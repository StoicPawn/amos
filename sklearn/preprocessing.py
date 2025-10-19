"""Minimal subset of ``sklearn.preprocessing`` used in the project."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .base import BaseEstimator


@dataclass
class StandardScaler(BaseEstimator):
    """Lightweight implementation of :class:`sklearn`'s ``StandardScaler``."""

    with_mean: bool = True
    with_std: bool = True

    def fit(self, X: np.ndarray) -> "StandardScaler":  # noqa: N803 (sklearn compat)
        matrix = self._as_2d(X)
        rows = matrix.to_list()
        n_samples = len(rows)
        n_features = len(rows[0]) if rows else 0

        if n_samples == 0 or n_features == 0:
            self.mean_ = [0.0] * n_features
            self.var_ = [1.0] * n_features
            self.scale_ = [1.0] * n_features
            self.n_features_in_ = n_features
            return self

        means: List[float] = []
        variances: List[float] = []
        for col in range(n_features):
            values = [float(row[col]) for row in rows]
            mean_val = sum(values) / len(values)
            if self.with_mean:
                means.append(mean_val)
            else:
                means.append(0.0)
            if self.with_std:
                variance = sum((val - mean_val) ** 2 for val in values) / len(values)
                variances.append(variance)
            else:
                variances.append(1.0)

        scales = [math.sqrt(var) if var > 0 else 1.0 for var in variances]

        self.mean_ = means
        self.var_ = variances
        self.scale_ = scales
        self.n_features_in_ = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 (sklearn compat)
        matrix = self._as_2d(X)
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("StandardScaler instance is not fitted")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )

        rows = matrix.to_list()
        transformed: List[List[float]] = []
        for row in rows:
            new_row: List[float] = []
            for idx, value in enumerate(row):
                adjusted = float(value)
                if self.with_mean:
                    adjusted -= self.mean_[idx]
                if self.with_std:
                    adjusted /= self.scale_[idx]
                new_row.append(adjusted)
            transformed.append(new_row)
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 (sklearn compat)
        return self.fit(X).transform(X)

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:  # noqa: N803 (sklearn compat)
        if hasattr(X, "to_numpy"):
            base = X.to_numpy().tolist()
            matrix = np.asarray(base, dtype=float)
        elif hasattr(X, "values"):
            values = X.values
            base = values.to_list() if hasattr(values, "to_list") else values
            matrix = np.asarray(base, dtype=float)
        elif hasattr(X, "tolist") and not isinstance(X, (list, tuple)):
            matrix = np.asarray(X.tolist(), dtype=float)
        else:
            matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            flattened = matrix.flatten()
            matrix = np.asarray([[val] for val in flattened], dtype=float)
        if matrix.ndim != 2:
            raise ValueError("StandardScaler expects 2D input")
        return matrix
