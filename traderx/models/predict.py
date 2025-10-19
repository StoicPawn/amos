"""CLI for inference generating target weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from traderx.features.pipelines import registry
from traderx.utils.config import load_config
from traderx.utils.io import atomic_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate targets from trained model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    clf = joblib.load(model_path / "classifier.joblib")
    reg = joblib.load(model_path / "regressor.joblib")

    features_df = pd.read_parquet(args.features)
    feature_funcs = registry()
    missing = set(feature_funcs) - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    proba = clf.predict_proba(features_df)[:, 1]
    payoff = reg.predict(features_df)
    weights = proba * payoff
    weights = weights / (abs(weights).sum() or 1)

    risk_cfg = load_config(args.risk).get("risk", {})
    max_weight = risk_cfg.get("max_symbol_weight", 0.1)
    weights = weights.clip(-max_weight, max_weight)

    df = pd.DataFrame({"symbol": features_df.index, "target_weight": weights})
    atomic_to_csv(df, args.out)


if __name__ == "__main__":
    main()
