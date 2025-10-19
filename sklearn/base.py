"""Simple compatibility helpers mirroring ``sklearn.base``."""
from __future__ import annotations


class BaseEstimator:
    """Minimal marker class."""

    def get_params(self):  # pragma: no cover - helper for completeness
        return {}


class ClassifierMixin(BaseEstimator):
    pass


class RegressorMixin(BaseEstimator):
    pass
