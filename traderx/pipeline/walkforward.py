"""Walk-forward training and evaluation pipeline."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from traderx.backtest.cv import PurgedKFold
from traderx.backtest.metrics import (
    estimate_slippage,
    expected_shortfall,
    max_drawdown,
    portfolio_turnover,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from traderx.features.pipelines import FeaturePipeline, registry as feature_registry
from traderx.labeling.triple_barrier import apply_triple_barrier, results_to_frame
from traderx.utils.io import AtomicWriter

_METRIC_COLUMN_MAP = {
    "Sharpe": "sharpe",
    "Sortino": "sortino",
    "PSR": "psr",
    "MaxDD": "max_drawdown",
    "ES95": "expected_shortfall_95",
    "Turnover": "turnover",
    "Slippage": "slippage",
}


@dataclass
class WalkForwardConfig:
    """Configuration bundle for the walk-forward loop."""

    retrain_freq: pd.Timedelta
    test_window: pd.Timedelta
    embargo: pd.Timedelta
    kfold: int
    metrics: List[str]
    costs_bps: float = 2.0

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WalkForwardConfig":
        retrain_freq_days = int(data.get("retrain_freq_days", 20))
        test_window_days = int(data.get("test_window_days", 20))
        embargo_bars = int(data.get("embargo_bars", 10))
        kfold = int(data.get("kfold", 5))
        metrics = list(data.get("metrics", list(_METRIC_COLUMN_MAP)))
        costs_bps = float(data.get("costs_bps", 2.0))
        return cls(
            retrain_freq=pd.Timedelta(days=retrain_freq_days),
            test_window=pd.Timedelta(days=test_window_days),
            embargo=pd.Timedelta(days=embargo_bars),
            kfold=kfold,
            metrics=metrics,
            costs_bps=costs_bps,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "retrain_freq_days": int(self.retrain_freq.days),
            "test_window_days": int(self.test_window.days),
            "embargo_bars": int(self.embargo.days),
            "kfold": self.kfold,
            "metrics": list(self.metrics),
            "costs_bps": float(self.costs_bps),
        }


@dataclass
class WalkForwardWindow:
    """Description of a single train/calibration/test split."""

    window_id: int
    train_index: pd.DatetimeIndex
    calib_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


@dataclass
class WalkForwardRunResult:
    """Aggregate outputs of a walk-forward execution."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: Dict[str, object]
    report_csv: Path | None
    report_parquet: Path | None
    summary_json: Path | None
    targets_csv: Path | None
    run_dir: Path
    active_model_dir: Path | None


@dataclass
class _WindowRunResult:
    metrics: Dict[str, object]
    predictions: pd.DataFrame
    calibration_method: str | None
    artifact_dir: Path


class _ConstantProbabilityClassifier(ClassifierMixin):
    """Fallback classifier when only one class is observed."""

    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def fit(self, X: Sequence[object], y: Sequence[object]) -> "_ConstantProbabilityClassifier":
        return self

    def predict_proba(self, X: Sequence[object]) -> np.ndarray:  # noqa: N803
        n_obs = len(X)
        proba = np.full(n_obs, self.probability)
        return np.column_stack([1 - proba, proba])


class _ConstantRegressor:
    """Fallback regressor returning a constant value."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, X: Sequence[object], y: Sequence[object]) -> "_ConstantRegressor":
        return self

    def predict(self, X: Sequence[object]) -> np.ndarray:  # noqa: N803
        return np.full(len(X), self.value)


class WalkForwardRunner:
    """Coordinate feature extraction, model training and evaluation."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: WalkForwardConfig,
        model_cfg: Mapping[str, object],
        artifact_root: Path,
    ) -> None:
        self.prices = prices.sort_index()
        self.config = config
        self.model_cfg = model_cfg
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self._create_run_directory()

        self._feature_names = self._resolve_feature_list(model_cfg)
        self._classifier_params = dict(model_cfg.get("lgbm", {}))
        self._classifier_params.setdefault("objective", "binary")
        self._regressor_params = dict(model_cfg.get("lgbm_regressor", self._classifier_params))
        self._regressor_params.setdefault("objective", "regression")

        self.features = self._compute_features()
        self.labels = self._compute_labels()
        aligned_labels = self.labels.reindex(self.features.index).ffill().bfill().fillna(0.0)
        self.labels = aligned_labels
        self.index = self.features.index
        self.windows = self._generate_windows()

    # ------------------------------------------------------------------
    def run(self) -> WalkForwardRunResult:
        self._materialize_run_configs()
        metrics_rows: List[Dict[str, object]] = []
        prediction_frames: List[pd.DataFrame] = []
        calibration_choices: List[str] = []
        last_window_id: int | None = None
        artifact_lookup: Dict[int, Path] = {}

        for window in self.windows:
            window_result = self._run_window(window)
            metrics_rows.append(window_result.metrics)
            prediction_frames.append(window_result.predictions)
            if window_result.calibration_method:
                calibration_choices.append(window_result.calibration_method)
            last_window_id = window.window_id
            artifact_lookup[window.window_id] = window_result.artifact_dir

        metrics_df = pd.DataFrame(metrics_rows)
        predictions_df = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["window_id", "timestamp", "probability", "expected_payoff", "weight", "actual_return"])
        )

        summary = self._build_summary(metrics_df, calibration_choices)
        active_model_dir, selection_info = self._select_active_model(metrics_df, artifact_lookup)
        if selection_info:
            summary["selection_metric"] = selection_info
        if active_model_dir is not None:
            summary["active_model"] = {
                "path": str(active_model_dir),
                "window_id": selection_info.get("window_id") if selection_info else None,
            }
        report_csv, report_parquet = self._write_report(metrics_df)
        summary_json = self._write_summary(summary)
        targets_csv = self._write_targets(predictions_df, last_window_id)

        return WalkForwardRunResult(
            metrics=metrics_df,
            predictions=predictions_df,
            summary=summary,
            report_csv=report_csv,
            report_parquet=report_parquet,
            summary_json=summary_json,
            targets_csv=targets_csv,
            run_dir=self.run_dir,
            active_model_dir=active_model_dir,
        )

    # ------------------------------------------------------------------
    def _run_window(self, window: WalkForwardWindow) -> _WindowRunResult:
        window_dir = self.run_dir / f"window_{window.window_id:03d}"
        window_dir.mkdir(parents=True, exist_ok=True)

        X_train = self.features.loc[window.train_index]
        y_train = (self.labels.loc[window.train_index, "label"] > 0).astype(int)
        reg_target = self.labels.loc[window.train_index, "ret"]

        X_calib = self.features.loc[window.calib_index]
        y_calib = (self.labels.loc[window.calib_index, "label"] > 0).astype(int)

        X_test = self.features.loc[window.test_index]
        test_labels = self.labels.loc[window.test_index]

        scaler = StandardScaler()
        if not X_train.empty:
            scaler.fit(X_train)
            X_train = self._array_to_frame(
                scaler.transform(X_train), X_train.index, X_train.columns
            )
            if not X_calib.empty:
                X_calib = self._array_to_frame(
                    scaler.transform(X_calib), X_calib.index, X_calib.columns
                )
            if not X_test.empty:
                X_test = self._array_to_frame(
                    scaler.transform(X_test), X_test.index, X_test.columns
                )

        joblib.dump(scaler, window_dir / "scaler.joblib")

        classifier, calibration_method = self._fit_classifier(X_train, y_train, X_calib, y_calib)
        regressor = self._fit_regressor(X_train, reg_target)

        joblib.dump(classifier, window_dir / "classifier.joblib")
        joblib.dump(regressor, window_dir / "regressor.joblib")

        metadata = {
            "window_id": window.window_id,
            "train_start": window.train_index.min(),
            "train_end": window.train_index.max(),
            "calibration_start": window.calib_index.min(),
            "calibration_end": window.calib_index.max(),
            "test_start": window.test_index.min(),
            "test_end": window.test_index.max(),
            "calibration_method": calibration_method,
            "features": list(self.features.columns),
        }
        with AtomicWriter(window_dir / "metadata.json") as fh:
            json.dump(metadata, fh, default=str)

        probabilities = classifier.predict_proba(X_test)[:, 1]
        payoff_pred = regressor.predict(X_test)
        weights = probabilities * payoff_pred
        returns = weights * test_labels["ret"].to_numpy()
        equity = (1 + returns).cumprod()

        turnover = portfolio_turnover(weights)
        slippage = estimate_slippage(turnover, self.config.costs_bps)

        metrics = {
            "window_id": window.window_id,
            "train_start": window.train_index.min(),
            "train_end": window.train_index.max(),
            "test_start": window.test_index.min(),
            "test_end": window.test_index.max(),
            "sharpe": sharpe_ratio(returns),
            "sortino": sortino_ratio(returns),
            "psr": probabilistic_sharpe_ratio(returns),
            "max_drawdown": max_drawdown(equity),
            "expected_shortfall_95": expected_shortfall(returns, alpha=0.95),
            "turnover": turnover,
            "slippage": slippage,
        }

        predictions = pd.DataFrame(
            {
                "window_id": window.window_id,
                "timestamp": window.test_index,
                "probability": probabilities,
                "expected_payoff": payoff_pred,
                "weight": weights,
                "actual_return": test_labels["ret"].to_numpy(),
            }
        )

        return _WindowRunResult(
            metrics=metrics,
            predictions=predictions,
            calibration_method=calibration_method,
            artifact_dir=window_dir,
        )

    # ------------------------------------------------------------------
    def _build_summary(
        self,
        metrics_df: pd.DataFrame,
        calibration_choices: Sequence[str],
    ) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "config": self.config.to_dict(),
            "windows": len(self.windows),
            "run_dir": str(self.run_dir),
        }
        if not metrics_df.empty:
            metrics_df = metrics_df.copy()
            metrics_df = metrics_df.sort_values("window_id")
            metric_summary: Dict[str, float] = {}
            for metric_name in self.config.metrics:
                column = _METRIC_COLUMN_MAP.get(metric_name)
                if column and column in metrics_df:
                    metric_summary[metric_name] = float(metrics_df[column].mean())
            summary["metrics"] = metric_summary
        else:
            summary["metrics"] = {}

        if calibration_choices:
            counts: Dict[str, int] = {}
            for method in calibration_choices:
                counts[method] = counts.get(method, 0) + 1
            summary["calibration_usage"] = counts
        else:
            summary["calibration_usage"] = {}
        return summary

    def _write_report(
        self,
        metrics_df: pd.DataFrame,
    ) -> tuple[Path | None, Path | None]:
        report_csv = self.run_dir / "report.csv"
        report_parquet = self.run_dir / "report.parquet"

        if not metrics_df.empty:
            with AtomicWriter(report_csv) as fh:
                metrics_df.to_csv(fh, index=False)
            try:
                metrics_df.to_parquet(report_parquet)
            except (ImportError, ValueError):
                report_parquet = None
        else:
            report_csv = None
            report_parquet = None
        return report_csv, report_parquet

    def _write_summary(self, summary: Dict[str, object]) -> Path | None:
        summary_path = self.run_dir / "summary.json"
        with AtomicWriter(summary_path) as fh:
            json.dump(summary, fh, default=str, indent=2)
        return summary_path

    def _write_targets(
        self,
        predictions_df: pd.DataFrame,
        last_window_id: int | None,
    ) -> Path | None:
        if predictions_df.empty or last_window_id is None:
            return None
        latest = predictions_df[predictions_df["window_id"] == last_window_id]
        if latest.empty:
            return None
        targets_path = self.run_dir / "targets.csv"
        with AtomicWriter(targets_path) as fh:
            latest.to_csv(fh, index=False)
        return targets_path

    # ------------------------------------------------------------------
    def _compute_features(self) -> pd.DataFrame:
        if not self._feature_names:
            pipeline = FeaturePipeline(self.prices)
            features = pipeline.compute()
            features.index = self.prices.index
            return features.dropna()

        registry = feature_registry()
        feature_frames: Dict[str, pd.Series] = {}
        for name in self._feature_names:
            func = registry.get(name)
            if func is None:
                raise KeyError(f"Unknown feature '{name}' requested in configuration")
            feature_frames[name] = pd.Series(func(self.prices), index=self.prices.index)
        features = pd.DataFrame(feature_frames, index=self.prices.index)
        return features.dropna()

    def _compute_labels(self) -> pd.DataFrame:
        label_cfg = self.model_cfg.get("labels", {})
        close = self.prices["close"]
        pt_mult = float(label_cfg.get("pt_mult", 1.5))
        sl_mult = float(label_cfg.get("sl_mult", 1.0))
        max_h = int(label_cfg.get("max_h", 20))
        volatility = close.pct_change().rolling(window=20).std().bfill().fillna(0.01)
        results = apply_triple_barrier(
            close.tolist(),
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_h=max_h,
            volatility=volatility.tolist(),
        )
        return results_to_frame(results, index=self.prices.index)

    def _generate_windows(self) -> List[WalkForwardWindow]:
        index = pd.DatetimeIndex(self.index)
        windows: List[WalkForwardWindow] = []
        if index.empty:
            return windows

        start = index[0] + self.config.test_window
        window_id = 0
        while start <= index[-1]:
            test_end = start + self.config.test_window
            test_index = index[(index >= start) & (index < test_end)]
            if len(test_index) == 0:
                break

            train_cutoff = start - self.config.embargo
            train_index = index[index < train_cutoff]
            if len(train_index) < self.config.kfold:
                start += self.config.retrain_freq
                continue

            pkf = PurgedKFold(n_splits=self.config.kfold, embargo_td=self.config.embargo)
            try:
                train_pos, calib_pos = next(pkf.split(train_index))
            except ValueError:
                start += self.config.retrain_freq
                continue

            train_subset = train_index.take(train_pos)
            calib_subset = train_index.take(calib_pos)
            if len(train_subset) == 0 or len(calib_subset) == 0:
                start += self.config.retrain_freq
                continue

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_index=pd.DatetimeIndex(train_subset),
                    calib_index=pd.DatetimeIndex(calib_subset),
                    test_index=pd.DatetimeIndex(test_index),
                )
            )
            window_id += 1
            start += self.config.retrain_freq

        return windows

    # ------------------------------------------------------------------
    def _fit_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> tuple[ClassifierMixin, str | None]:
        if X_train.empty:
            return _ConstantProbabilityClassifier(0.5), None

        y_train_values = y_train.to_numpy()
        y_calib_values = y_calib.to_numpy()

        if np.unique(y_train_values).size < 2 or np.unique(y_calib_values).size < 2:
            probability = float(y_train_values.mean()) if y_train_values.size else 0.5
            return _ConstantProbabilityClassifier(probability), None

        methods = ["isotonic", "sigmoid"]
        candidates: List[tuple[float, str, ClassifierMixin]] = []
        for method in methods:
            if method == "isotonic" and len(X_calib) < 3:
                continue
            estimator = LGBMClassifier(**self._classifier_params)
            estimator.fit(X_train, y_train_values)
            calibrator = CalibratedClassifierCV(estimator, method=method, cv="prefit")
            calibrator.fit(X_calib, y_calib_values)
            proba = calibrator.predict_proba(X_calib)[:, 1]
            brier = np.mean((proba - y_calib_values) ** 2)
            candidates.append((brier, method, calibrator))

        if not candidates:
            estimator = LGBMClassifier(**self._classifier_params)
            estimator.fit(X_train, y_train_values)
            return estimator, None

        candidates.sort(key=lambda item: item[0])
        _, method, calibrator = candidates[0]
        return calibrator, method

    def _fit_regressor(self, X_train: pd.DataFrame, y_train: pd.Series):
        if X_train.empty:
            return _ConstantRegressor(0.0)
        y_values = y_train.to_numpy()
        if np.allclose(y_values, y_values[0]):
            return _ConstantRegressor(float(y_values[0]))
        regressor = LGBMRegressor(**self._regressor_params)
        regressor.fit(X_train, y_values)
        return regressor

    @staticmethod
    def _array_to_frame(array: object, index: pd.Index, columns: Sequence[str]) -> pd.DataFrame:
        if hasattr(array, "to_list"):
            rows = array.to_list()
        elif hasattr(array, "tolist"):
            rows = array.tolist()
        else:
            rows = list(array)
        data = {col: [] for col in columns}
        if not rows:
            return pd.DataFrame(data, index=index)
        for row in rows:
            if isinstance(row, (list, tuple)):
                iterable = row
            else:
                iterable = [row]
            for idx, col in enumerate(columns):
                data[col].append(iterable[idx])
        return pd.DataFrame(data, index=index)

    @staticmethod
    def _resolve_feature_list(model_cfg: Mapping[str, object]) -> List[str]:
        features_cfg = model_cfg.get("features", {})
        if isinstance(features_cfg, Mapping):
            feature_list = features_cfg.get("list", [])
        else:
            feature_list = []
        return list(feature_list)

    # ------------------------------------------------------------------
    def _create_run_directory(self) -> Path:
        existing = [
            path
            for path in self.artifact_root.iterdir()
            if path.is_dir() and path.name.startswith("run_")
        ]
        next_idx = 0
        if existing:
            indices = []
            for path in existing:
                try:
                    indices.append(int(path.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if indices:
                next_idx = max(indices) + 1
        run_dir = self.artifact_root / f"run_{next_idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _materialize_run_configs(self) -> None:
        config_path = self.run_dir / "walkforward_config.json"
        model_path = self.run_dir / "model_config.json"
        with AtomicWriter(config_path) as fh:
            json.dump(self.config.to_dict(), fh, indent=2)
        with AtomicWriter(model_path) as fh:
            json.dump(self._sanitize_model_config(self.model_cfg), fh, indent=2, default=str)

    def _sanitize_model_config(self, config: Mapping[str, object]) -> Dict[str, object]:
        serializable: Dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, Mapping):
                serializable[key] = self._sanitize_model_config(value)
            elif isinstance(value, (list, tuple)):
                serializable[key] = [self._convert_scalar(v) for v in value]
            else:
                serializable[key] = self._convert_scalar(value)
        return serializable

    @staticmethod
    def _convert_scalar(value: object) -> object:
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def _select_active_model(
        self,
        metrics_df: pd.DataFrame,
        artifact_lookup: Mapping[int, Path],
    ) -> tuple[Path | None, Dict[str, object]]:
        if metrics_df.empty:
            return None, {}
        selection_metric = self._resolve_selection_metric(metrics_df)
        if selection_metric is None:
            return None, {}
        metric_name, column, higher_is_better = selection_metric
        metrics_df = metrics_df.dropna(subset=[column])
        if metrics_df.empty:
            return None, {}
        sorted_df = metrics_df.sort_values(column, ascending=not higher_is_better)
        best_row = sorted_df.iloc[0]
        window_id = int(best_row["window_id"])
        summary_entry = {
            "metric": metric_name,
            "value": float(best_row[column]),
            "window_id": window_id,
            "direction": "max" if higher_is_better else "min",
        }
        artifact_dir = artifact_lookup.get(window_id)
        if artifact_dir is None:
            return None, summary_entry
        active_dir = self.run_dir / "active"
        active_dir.mkdir(exist_ok=True)
        for artifact_name in ("classifier.joblib", "regressor.joblib", "metadata.json"):
            source = artifact_dir / artifact_name
            if source.exists():
                shutil.copy2(source, active_dir / artifact_name)
        if (artifact_dir / "metadata.json").exists():
            with AtomicWriter(active_dir / "selection.json") as fh:
                json.dump(dict(summary_entry), fh, indent=2)
        return active_dir, summary_entry

    def _resolve_selection_metric(
        self,
        metrics_df: pd.DataFrame,
    ) -> tuple[str, str, bool] | None:
        for metric_name in self.config.metrics:
            column = _METRIC_COLUMN_MAP.get(metric_name)
            if column and column in metrics_df.columns:
                higher_is_better = metric_name not in {"Turnover", "Slippage"}
                if metric_name == "MaxDD":
                    higher_is_better = True
                elif metric_name == "ES95":
                    higher_is_better = True
                return metric_name, column, higher_is_better
        return None
