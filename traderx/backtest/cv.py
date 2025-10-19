"""Cross-validation utilities with purging and embargo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generator, Iterable, List, Sequence, Tuple


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


def _as_timestamp(value: object) -> datetime:
    """Return ``value`` as a timezone aware :class:`datetime`."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise TypeError(f"Cannot interpret {value!r} as a timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    raise TypeError(f"Unsupported timestamp value: {value!r}")
