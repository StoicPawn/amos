"""Light-weight linear models used in the kata."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .base import ClassifierMixin, RegressorMixin


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


class LogisticRegression(ClassifierMixin):
    """Very small logistic regression returning constant probabilities."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.max_iter = max_iter
        self._probability = 0.5

    def fit(self, X: Iterable[Sequence[float]], y: Iterable[float]) -> "LogisticRegression":
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


class LinearRegression(RegressorMixin):
    """Linear regression approximated with a simple mean predictor."""

    def __init__(self) -> None:
        self._value = 0.0

    def fit(self, X: Iterable[Sequence[float]], y: Iterable[float]) -> "LinearRegression":
        values = _ensure_vector(y)
        if values:
            self._value = sum(values) / len(values)
        else:
            self._value = 0.0
        return self

    def predict(self, X: Iterable[Sequence[float]]) -> np.NDArray:  # noqa: N803
        matrix = _ensure_matrix(X)
        return np.full(len(matrix), self._value)
