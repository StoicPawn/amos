"""Walk-forward backtest CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from traderx.pipeline import WalkForwardConfig, WalkForwardRunner
from traderx.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--walkforward", required=True, help="YAML with walk-forward settings")
    parser.add_argument("--model", required=True, help="YAML with model configuration")
    parser.add_argument("--out", required=True, help="Directory for generated artifacts")
    parser.add_argument("--prices", help="CSV with OHLC data (index column must be datetime)")
    parser.add_argument("--costs-bps", type=float, dest="costs_bps", help="Override execution costs in basis points")
    return parser.parse_args()


def _load_prices(path: str | None) -> pd.DataFrame:
    if path is None:
        idx = pd.date_range("2022-01-01", periods=200, freq="B")
        prices = pd.DataFrame(
            {
                "close": 100 + np.sin(np.linspace(0, 12, len(idx))) * 5 + np.linspace(0, 1, len(idx)),
                "high": 100 + np.sin(np.linspace(0, 12, len(idx))) * 5.5 + np.linspace(0, 1, len(idx)),
                "low": 100 + np.sin(np.linspace(0, 12, len(idx))) * 4.5 + np.linspace(0, 1, len(idx)),
            },
            index=idx,
        )
        return prices
    prices = pd.read_csv(path, index_col=0, parse_dates=[0])
    required_cols = {"close", "high", "low"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Price file {path} missing columns: {sorted(missing)}")
    return prices.sort_index()


def main() -> None:
    args = parse_args()
    wf_bundle = load_config(args.walkforward)
    wf_section = wf_bundle.get("walkforward", {})
    config = WalkForwardConfig.from_dict(wf_section)
    if args.costs_bps is not None:
        config.costs_bps = float(args.costs_bps)

    model_bundle = load_config(args.model)
    model_cfg = model_bundle.data

    prices = _load_prices(args.prices)
    runner = WalkForwardRunner(prices, config, model_cfg, Path(args.out))
    result = runner.run()
    if result.report_csv:
        print(f"Walk-forward report saved to {result.report_csv}")
    if result.summary_json:
        print(f"Summary saved to {result.summary_json}")
    if result.targets_csv:
        print(f"Targets saved to {result.targets_csv}")


if __name__ == "__main__":
    main()
