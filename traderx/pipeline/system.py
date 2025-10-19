"""End-to-end trading system assembly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from traderx.backtest.engine import BacktestEngine, BacktestResult
from traderx.features.pipelines import FeaturePipeline
from traderx.labeling.triple_barrier import apply_triple_barrier, results_to_frame
from traderx.portfolio.risk import apply_vol_target, enforce_gross_limits


@dataclass
class SymbolArtifacts:
    """Artifacts generated for a single symbol during a system run."""

    features: pd.DataFrame
    labels: pd.DataFrame
    probabilities: pd.Series
    payoff: pd.Series
    raw_weights: pd.Series


@dataclass
class TradingRunResult:
    """Container describing the outcome of a :class:`TradingSystem` run."""

    weights: pd.DataFrame
    backtest: BacktestResult
    artifacts: Dict[str, SymbolArtifacts]


class _ConstantClassifier:
    """Fallback classifier when only a single class is observed."""

    def __init__(self, probability: float) -> None:
        self._probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, X: object) -> np.ndarray:  # noqa: N803  (sklearn compat)
        n_obs = len(X)
        prob = np.full(n_obs, self._probability)
        return np.column_stack([1 - prob, prob])


class _ConstantRegressor:
    """Fallback regressor that always returns a constant value."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def predict(self, X: object) -> np.ndarray:  # noqa: N803 (sklearn compat)
        return np.full(len(X), self._value)


class TradingSystem:
    """End-to-end orchestration for signal generation and backtesting."""

    def __init__(
        self,
        price_panel: Mapping[str, pd.DataFrame],
        risk_cfg: Mapping[str, float] | None = None,
        *,
        costs_bps: float = 2.0,
        barrier_cfg: Mapping[str, float] | None = None,
    ) -> None:
        self._prices: Dict[str, pd.DataFrame] = {}
        index: pd.Index | None = None
        for symbol, df in price_panel.items():
            if {"close", "high", "low"} - set(df.columns):
                missing = {"close", "high", "low"} - set(df.columns)
                raise ValueError(f"Missing columns {missing} for symbol {symbol}")
            ordered = df.sort_index()
            if index is None:
                index = ordered.index
            else:
                index = index.union(ordered.index)
            self._prices[symbol] = ordered
        if index is None:
            raise ValueError("price_panel must contain at least one symbol")
        self._index = index.sort_values()
        for symbol, df in self._prices.items():
            aligned = df.reindex(self._index).ffill().bfill()
            self._prices[symbol] = aligned.astype(float)
        self._risk_cfg = dict(risk_cfg or {})
        self._barrier_cfg = dict(barrier_cfg or {})
        self._costs_bps = costs_bps

    def run(self) -> TradingRunResult:
        """Execute the full research workflow for the configured universe."""

        artifacts: Dict[str, SymbolArtifacts] = {}
        weights_map: Dict[str, pd.Series] = {}

        for symbol, prices in self._prices.items():
            symbol_artifacts, symbol_weights = self._process_symbol(symbol, prices)
            artifacts[symbol] = symbol_artifacts
            weights_map[symbol] = symbol_weights

        weights_df = (
            pd.DataFrame(weights_map, index=self._index)
            .fillna(0.0)
            .sort_index(axis=1)
        )

        max_gross = self._risk_cfg.get("max_gross")
        if max_gross is not None:
            for ts in weights_df.index:
                weights_df.loc[ts] = enforce_gross_limits(weights_df.loc[ts], float(max_gross))

        price_lists = {symbol: prices["close"].tolist() for symbol, prices in self._prices.items()}
        weight_lists = {symbol: weights_df[symbol].tolist() for symbol in weights_df.columns}
        engine = BacktestEngine(price_lists, costs_bps=self._costs_bps)
        backtest = engine.run(weight_lists)

        return TradingRunResult(weights=weights_df, backtest=backtest, artifacts=artifacts)

    # ------------------------------------------------------------------
    def _process_symbol(self, symbol: str, prices: pd.DataFrame) -> Tuple[SymbolArtifacts, pd.Series]:
        features = FeaturePipeline(prices).compute()
        features.index = prices.index
        features = features.dropna()
        if features.empty:
            zero_weights = pd.Series(0.0, index=self._index)
            empty = pd.DataFrame(index=self._index)
            artifacts = SymbolArtifacts(
                features=empty,
                labels=empty,
                probabilities=pd.Series(0.0, index=self._index),
                payoff=pd.Series(0.0, index=self._index),
                raw_weights=pd.Series(0.0, index=self._index),
            )
            return artifacts, zero_weights

        label_results = apply_triple_barrier(
            prices["close"].tolist(),
            pt_mult=float(self._barrier_cfg.get("pt_mult", 1.0)),
            sl_mult=float(self._barrier_cfg.get("sl_mult", 1.0)),
            max_h=int(self._barrier_cfg.get("max_h", 5)),
            volatility=self._compute_volatility(prices["close"]),
        )
        labels = results_to_frame(label_results, index=prices.index)
        labels = labels.reindex(features.index).ffill().bfill().fillna(0.0)

        if labels.empty:
            labels = pd.DataFrame({"label": np.zeros(len(features)), "ret": np.zeros(len(features))}, index=features.index)

        feature_matrix = features.astype(float).values
        classifier, regressor = self._fit_models(feature_matrix, labels)

        probabilities = classifier.predict_proba(feature_matrix)[:, 1]
        payoff = regressor.predict(feature_matrix)
        raw_weights = probabilities * payoff
        raw_weights_series = pd.Series(raw_weights, index=features.index).fillna(0.0)

        symbol_weights = self._apply_symbol_risk(raw_weights_series, prices)

        artifacts = SymbolArtifacts(
            features=features,
            labels=labels,
            probabilities=pd.Series(probabilities, index=features.index),
            payoff=pd.Series(payoff, index=features.index),
            raw_weights=pd.Series(raw_weights, index=features.index),
        )
        return artifacts, symbol_weights

    def _compute_volatility(self, close: pd.Series) -> Iterable[float]:
        lookback = int(self._barrier_cfg.get("vol_lookback", 10))
        vol = close.pct_change().rolling(lookback).std().bfill().ffill()
        vol = vol.replace(0, vol[vol > 0].min() or 0.01)
        if vol.isna().any():
            vol = vol.fillna(0.01)
        return vol.tolist()

    def _fit_models(self, matrix: np.ndarray, labels: pd.DataFrame) -> Tuple[LogisticRegression | _ConstantClassifier, LinearRegression | _ConstantRegressor]:
        y_class = (labels["label"].astype(float) > 0).astype(int).values
        y_reg = labels["ret"].astype(float).values

        if matrix.size == 0:
            return _ConstantClassifier(0.0), _ConstantRegressor(0.0)

        unique = np.unique(y_class)
        if unique.size <= 1:
            classifier: LogisticRegression | _ConstantClassifier = _ConstantClassifier(float(unique[0] if unique.size else 0.0))
        else:
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(matrix, y_class)

        if np.allclose(y_reg, y_reg[0]):
            regressor: LinearRegression | _ConstantRegressor = _ConstantRegressor(float(y_reg[0] if y_reg.size else 0.0))
        else:
            regressor = LinearRegression()
            regressor.fit(matrix, y_reg)

        return classifier, regressor

    def _apply_symbol_risk(self, weights: pd.Series, prices: pd.DataFrame) -> pd.Series:
        full_weights = pd.Series(0.0, index=self._index)
        full_weights.loc[weights.index] = weights

        close = prices["close"].reindex(self._index).ffill().bfill()
        returns = close.pct_change().fillna(0.0)

        target_vol = self._risk_cfg.get("target_vol")
        if target_vol is not None:
            full_weights = apply_vol_target(full_weights, returns, float(target_vol))

        max_symbol_weight = self._risk_cfg.get("max_symbol_weight")
        if max_symbol_weight is not None:
            full_weights = full_weights.clip(-float(max_symbol_weight), float(max_symbol_weight))

        return full_weights.fillna(0.0)
