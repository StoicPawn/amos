"""Cross-validation utilities with purging and embargo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Iterable, List, Sequence, Tuple

import pandas as pd


@dataclass
class PurgedKFold:
    """Time-aware cross-validation with purging and embargo."""

    n_splits: int
    embargo_td: timedelta

    def split(
        self,
        X: Sequence[object],
        y: Iterable | None = None,
        groups: Iterable | None = None,
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        values = list(X)
        n = len(values)
        if n < self.n_splits:
            raise ValueError("Number of samples must be >= n_splits")

        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = list(range(start, stop))
            if not test_indices:
                continue

            test_start = _as_timestamp(values[test_indices[0]])
            test_end = _as_timestamp(values[test_indices[-1]])

            purge_start = test_start - self.embargo_td
            purge_end = test_end + self.embargo_td

            train_indices: List[int] = []
            for idx, value in enumerate(values):
                if idx in test_indices:
                    continue
                ts = _as_timestamp(value)
                if purge_start <= ts <= purge_end:
                    continue
                train_indices.append(idx)

            yield train_indices, test_indices
            current = stop


def _as_timestamp(value: object) -> pd.Timestamp:
    """Return ``value`` as :class:`pandas.Timestamp` for comparisons."""

    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)
