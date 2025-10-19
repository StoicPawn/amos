"""CLI for model training."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from traderx.features.pipelines import FeaturePipeline
from traderx.labeling.triple_barrier import apply_triple_barrier, results_to_frame
from traderx.utils.config import load_config
from traderx.utils.io import atomic_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TraderX model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--walkforward", required=True)
    parser.add_argument("--costs", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prices", required=False, help="CSV with columns close/high/low")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_config(args.config)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.prices:
        prices = pd.read_csv(args.prices, parse_dates=[0], index_col=0)
    else:
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "close": 100 + pd.Series(range(100), index=idx),
            "high": 101 + pd.Series(range(100), index=idx),
            "low": 99 + pd.Series(range(100), index=idx),
        })

    features = FeaturePipeline(prices).compute()
    features.index = prices.index
    features = features.bfill()
    labels_cfg = model_cfg.get("labels", {})
    label_results = apply_triple_barrier(
        prices["close"],
        pt_mult=labels_cfg.get("pt_mult", 1.5),
        sl_mult=labels_cfg.get("sl_mult", 1.0),
        max_h=labels_cfg.get("max_h", 20),
    )
    labels = results_to_frame(label_results, index=prices.index)
    labels = labels.reindex(features.index).ffill().bfill().fillna(0.0)
    y = labels["label"].values

    clf = GradientBoostingClassifier()
    reg = GradientBoostingRegressor()
    clf.fit(features, y)
    reg.fit(features, labels["ret"].values)

    joblib.dump(clf, out_path / "classifier.joblib")
    joblib.dump(reg, out_path / "regressor.joblib")
    atomic_to_csv(labels, out_path / "labels.csv")


if __name__ == "__main__":
    main()
