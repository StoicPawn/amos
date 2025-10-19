"""Placeholder LightGBM estimators used for testing."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _ensure_vector(y: Iterable[float]) -> list[float]:
    if isinstance(y, np.NDArray):
        return [float(v) for v in y.flatten()]
    return [float(v) for v in y]


def _ensure_matrix(X: Iterable[Sequence[float]]) -> list[list[float]]:
    if hasattr(X, "values"):
        array = X.values
        return [row.copy() for row in array._data]
    if isinstance(X, np.NDArray):
        return [list(row) for row in X._data]
    return [list(row) for row in X]


class LGBMClassifier:
    def __init__(self, **params) -> None:
        self.params = params
        self._probability = 0.5

    def fit(self, X: Iterable[Sequence[float]], y: Iterable[int]) -> "LGBMClassifier":
        values = _ensure_vector(y)
        if values:
            self._probability = max(0.0, min(1.0, sum(values) / len(values)))
        else:
            self._probability = 0.5
        return self

    def predict_proba(self, X: Iterable[Sequence[float]]) -> np.NDArray:  # noqa: N803
        matrix = _ensure_matrix(X)
        n_obs = len(matrix)
        proba = np.full(n_obs, self._probability)
        return np.column_stack([1 - proba, proba])


class LGBMRegressor:
    def __init__(self, **params) -> None:
        self.params = params
        self._value = 0.0

    def fit(self, X: Iterable[Sequence[float]], y: Iterable[float]) -> "LGBMRegressor":
        values = _ensure_vector(y)
        if values:
            self._value = sum(values) / len(values)
        else:
            self._value = 0.0
        return self

    def predict(self, X: Iterable[Sequence[float]]) -> np.NDArray:  # noqa: N803
        matrix = _ensure_matrix(X)
        return np.full(len(matrix), self._value)


__all__ = ["LGBMClassifier", "LGBMRegressor"]
