"""Probability calibration helpers."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin


def platt_scale(model: ClassifierMixin, X, y) -> CalibratedClassifierCV:
    """Return a Platt-scaled classifier."""
    return CalibratedClassifierCV(model, method="sigmoid").fit(X, y)


def isotonic_scale(model: ClassifierMixin, X, y) -> CalibratedClassifierCV:
    """Return an isotonic calibrated classifier."""
    return CalibratedClassifierCV(model, method="isotonic").fit(X, y)
