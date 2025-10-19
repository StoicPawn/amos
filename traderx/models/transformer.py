"""Placeholder transformer model (not implemented)."""
from __future__ import annotations


class DummyTransformer:
    """Simple placeholder class."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.mean(axis=1)
