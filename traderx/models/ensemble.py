"""Simple ensemble placeholder combining classifier and regressor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class SignalEnsemble:
    """Combine probability and payoff estimates into a score."""

    classifier: ClassifierMixin
    regressor: RegressorMixin

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SignalEnsemble":
        self.classifier.fit(X, y)
        self.regressor.fit(X, y)
        return self

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        proba = self.classifier.predict_proba(X)[:, 1]
        payoff = self.regressor.predict(X)
        score = proba * payoff
        return pd.Series(score, index=X.index)

    def predict_weights(self, X: pd.DataFrame, scaler: float = 1.0) -> pd.Series:
        score = self.predict_score(X)
        weights = score / np.nanstd(score) if np.nanstd(score) else score
        return (weights * scaler).clip(-1, 1)


def calibrate_probabilities(model: ClassifierMixin, X: pd.DataFrame, y: Iterable[int]) -> ClassifierMixin:
    """Placeholder for calibration (no-op)."""
    return model
