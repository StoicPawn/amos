"""Integration tests for the walk-forward runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traderx.pipeline import WalkForwardConfig, WalkForwardRunner
from traderx.utils.config import load_config


def _synthetic_prices(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.linspace(0, 10, n)
    close = 100 + np.sin(base) * 2 + base * 0.05
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


def test_walkforward_runner_generates_artifacts(tmp_path: Path) -> None:
    prices = _synthetic_prices()
    wf_cfg = load_config("configs/walkforward.yaml")
    config = WalkForwardConfig.from_dict(wf_cfg.get("walkforward", {}))
    model_cfg = load_config("configs/model_lgbm.yaml").data
    # Use lighter models for the test run
    model_cfg.setdefault("lgbm", {})
    model_cfg["lgbm"].update({"n_estimators": 50, "num_leaves": 31, "learning_rate": 0.1})

    runner = WalkForwardRunner(prices, config, model_cfg, tmp_path)
    result = runner.run()

    assert not result.metrics.empty
    run_dir = result.run_dir
    assert run_dir.exists()
    assert any(path.name.startswith("window_") for path in run_dir.iterdir())
    assert result.summary_json and result.summary_json.exists()
    assert result.report_csv and result.report_csv.exists()
    if result.report_parquet:
        assert result.report_parquet.exists()
    if result.targets_csv:
        assert result.targets_csv.exists()

    for window_dir in run_dir.glob("window_*/"):
        assert (window_dir / "classifier.joblib").exists()
        assert (window_dir / "regressor.joblib").exists()
        assert (window_dir / "scaler.joblib").exists()
        assert (window_dir / "metadata.json").exists()

    assert (run_dir / "walkforward_config.json").exists()
    assert (run_dir / "model_config.json").exists()

    if result.active_model_dir:
        assert (result.active_model_dir / "classifier.joblib").exists()
        assert (result.active_model_dir / "regressor.joblib").exists()
        assert "selection_metric" in result.summary

    summary_metrics = result.summary.get("metrics", {})
    for key in config.metrics:
        if key in summary_metrics:
            assert isinstance(summary_metrics[key], float)
