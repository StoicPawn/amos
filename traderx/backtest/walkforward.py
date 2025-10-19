"""Walk-forward backtest CLI."""
from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from traderx.backtest.cv import PurgedKFold
from traderx.backtest.engine import BacktestEngine
from traderx.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--exec", dest="exec_cfg", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wf_cfg = load_config(args.config).get("walkforward", {})
    prices = pd.DataFrame({
        "AAPL": [100, 101, 102, 103, 102, 104],
        "MSFT": [200, 201, 199, 202, 203, 205],
    })
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="D")
    prices.index = idx

    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights.iloc[:, 0] = 0.05
    engine = BacktestEngine(prices, costs_bps=2.0)
    result = engine.run(weights)
    result.to_csv("data/processed/walkforward_results.csv")

    embargo = timedelta(days=wf_cfg.get("embargo_bars", 10))
    pkf = PurgedKFold(n_splits=wf_cfg.get("kfold", 5), embargo_td=embargo)
    list(pkf.split(prices))


if __name__ == "__main__":
    main()
