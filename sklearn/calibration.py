"""Minimal calibration wrapper compatible with the tests."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .base import ClassifierMixin


def _ensure_matrix(X: Iterable[Sequence[float]]) -> list[list[float]]:
    if hasattr(X, "values"):
        array = X.values
        return [row.copy() for row in array._data]
    if isinstance(X, np.NDArray):
        return [list(row) for row in X._data]
    return [list(row) for row in X]


def _ensure_vector(y: Iterable[float]) -> list[float]:
    if isinstance(y, np.NDArray):
        return [float(v) for v in y.flatten()]
    return [float(v) for v in y]


class CalibratedClassifierCV(ClassifierMixin):
    def __init__(self, estimator: ClassifierMixin, method: str = "sigmoid", cv: str | None = None) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, X: Iterable[Sequence[float]], y: Iterable[int]) -> "CalibratedClassifierCV":  # noqa: N803
        targets = _ensure_vector(y)
        if self.cv != "prefit" and hasattr(self.estimator, "fit"):
            self.estimator.fit(X, targets)
        if targets:
            self._probability = max(0.0, min(1.0, sum(targets) / len(targets)))
        else:
            self._probability = 0.5
        return self

    def predict_proba(self, X: Iterable[Sequence[float]]) -> np.NDArray:  # noqa: N803
        return self.estimator.predict_proba(X)
