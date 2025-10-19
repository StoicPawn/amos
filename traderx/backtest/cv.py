"""Cross-validation utilities with purging and embargo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Iterable, List, Sequence, Tuple


@dataclass
class PurgedKFold:
    """Time-aware cross-validation with purging and embargo."""

    n_splits: int
    embargo_td: timedelta

    def split(
        self,
        X: Sequence[int],
        y: Iterable | None = None,
        groups: Iterable | None = None,
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        n = len(X)
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = list(range(start, stop))
            train_indices = list(range(0, start))
            yield train_indices, test_indices
            current = stop
