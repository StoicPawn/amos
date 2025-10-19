"""Extremely small stand-in for :mod:`pandas` used in the kata tests.

The full project leverages pandas heavily, however the kata runtime does
not ship with external dependencies.  Only a very small portion of the
API is required by the unit tests so this module implements a focused
subset that mimics the behaviour closely enough for the exercises.

The goal of this shim is fidelity over feature completeness: operations
are intentionally restricted to the patterns exercised in the tests
(series/DataFrame creation, simple arithmetic, forward/backward filling
and a handful of statistical helpers).  The implementation is written in
pure Python using standard library types.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

NA_VALUE = float("nan")


# ---------------------------------------------------------------------------
# Small utility helpers


def _is_na(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------------------------
# Index helpers


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed
        except ValueError:
            pass
    raise TypeError(f"Cannot convert {value!r} to datetime")


class Index:
    """Minimal index implementation backed by a Python list."""

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        self._data: List[Any] = list(values) if values is not None else []

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(self._data[item])
        if isinstance(item, Series):
            mask = [bool(v) for v in item.to_list()]
            return self.__class__([val for val, flag in zip(self._data, mask) if flag])
        if isinstance(item, list) and item and isinstance(item[0], bool):
            return self.__class__([val for val, flag in zip(self._data, item) if flag])
        return self._data[item]

    def __contains__(self, item: Any) -> bool:
        return item in self._data

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Index({self._data!r})"

    def to_list(self) -> List[Any]:
        return self._data.copy()

    def union(self, other: "Index") -> "Index":
        seen = {value: None for value in self._data}
        for value in other:
            seen.setdefault(value, None)
        return self.__class__(list(seen))

    def sort_values(self) -> "Index":
        return self.__class__(sorted(self._data))

    def take(self, positions: Sequence[int]) -> "Index":
        return self.__class__([self._data[i] for i in positions])

    def _compare(self, other, op) -> Series:
        if isinstance(other, Index):
            values = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            values = [op(a, other) for a in self._data]
        return Series(values, index=self._data)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def min(self):  # pragma: no cover - convenience for tests
        return min(self._data) if self._data else None

    def max(self):
        return max(self._data) if self._data else None

    @property
    def empty(self) -> bool:
        return not self._data


class DatetimeIndex(Index):
    def __init__(self, values: Iterable[Any] | None = None) -> None:
        if values is None:
            super().__init__([])
        else:
            super().__init__([_ensure_datetime(v) for v in values])

    def sort_values(self) -> "DatetimeIndex":
        return DatetimeIndex(sorted(self._data))


# ---------------------------------------------------------------------------
# Scalar helpers


class Timedelta:
    def __init__(self, days: int = 0) -> None:
        self._delta = timedelta(days=days)

    @property
    def days(self) -> int:
        return self._delta.days

    def to_pytimedelta(self) -> timedelta:
        return self._delta

    def __add__(self, other: Any):
        if isinstance(other, datetime):
            return other + self._delta
        raise TypeError("Timedelta can only be added to datetime objects")

    __radd__ = __add__

    def __rsub__(self, other: Any):
        if isinstance(other, datetime):
            return other - self._delta
        raise TypeError("Timedelta subtraction requires a datetime operand")


Timestamp = datetime  # simple alias for compatibility


# ---------------------------------------------------------------------------
# Series implementation


class Series:
    def __init__(self, data: Iterable[Any] | Mapping[Any, Any] | None = None, index: Iterable[Any] | None = None, name: str | None = None) -> None:
        if isinstance(data, Mapping):
            if index is None:
                index = list(data.keys())
            values = [data.get(key) for key in index]
        elif data is None:
            values = []
            index = index or []
        else:
            if isinstance(data, np.NDArray):
                values = data.flatten()
            elif isinstance(data, (int, float, bool)):
                target_len = len(index) if index is not None else 1
                values = [data] * target_len
            else:
                try:
                    values = list(data)
                except TypeError:
                    values = [data]
            if index is None:
                index = list(range(len(values)))
        if index is None:
            index = []
        if len(values) != len(index):
            raise ValueError("Length mismatch between data and index")
        self._data = [self._coerce_value(v) for v in values]
        self._index = Index(index)
        self.name = name

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return float(value)
        if value is None:
            return NA_VALUE
        return value

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            index = self._index[key]
            return Series(self._data[key], index=index)
        if isinstance(key, Series):
            if len(key) != len(self):
                raise ValueError("Boolean index length mismatch")
            selected = [self._data[i] for i, flag in enumerate(key._data) if flag]
            new_index = [self._index._data[i] for i, flag in enumerate(key._data) if flag]
            return Series(selected, index=new_index)
        if isinstance(key, (list, tuple, Index)):
            indices = [self._index._data.index(k) for k in key]
            return Series([self._data[i] for i in indices], index=[self._index._data[i] for i in indices])
        if isinstance(key, list) and key and isinstance(key[0], bool):
            selected = [self._data[i] for i, flag in enumerate(key) if flag]
            new_index = [self._index._data[i] for i, flag in enumerate(key) if flag]
            return Series(selected, index=new_index)
        if key in self._index:
            pos = self._index._data.index(key)
            return self._data[pos]
        if isinstance(key, int):
            return self._data[key]
        raise KeyError(key)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, (list, tuple, Index, Series)):
            if isinstance(key, Series):
                keys = key.to_list()
            elif isinstance(key, Index):
                keys = key.to_list()
            else:
                keys = list(key)
            if isinstance(value, Series):
                values = value.to_list()
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value] * len(keys)
            if len(keys) != len(values):
                raise ValueError("Length mismatch during assignment")
            for item, val in zip(keys, values):
                self[item] = val
            return
        if key in self._index:
            pos = self._index._data.index(key)
            self._data[pos] = self._coerce_value(value)
        else:
            self._index._data.append(key)
            self._data.append(self._coerce_value(value))

    # arithmetic -------------------------------------------------------
    def _binary_op(self, other, op) -> "Series":
        if isinstance(other, Series):
            if len(other) != len(self):
                raise ValueError("Series length mismatch")
            values = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            values = [op(a, other) for a in self._data]
        return Series(values, index=self._index._data)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b if b != 0 else 0.0)

    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    # comparisons -----------------------------------------------------
    def _compare(self, other, op) -> "Series":
        if isinstance(other, Series):
            values = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            values = [op(a, other) for a in self._data]
        return Series(values, index=self._index._data)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __eq__(self, other):  # type: ignore[override]
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):  # type: ignore[override]
        return self._compare(other, lambda a, b: a != b)

    def __and__(self, other):
        if isinstance(other, Series):
            values = [bool(a) and bool(b) for a, b in zip(self._data, other._data)]
        else:
            values = [bool(a) and bool(other) for a in self._data]
        return Series(values, index=self._index._data)

    def __or__(self, other):
        if isinstance(other, Series):
            values = [bool(a) or bool(b) for a, b in zip(self._data, other._data)]
        else:
            values = [bool(a) or bool(other) for a in self._data]
        return Series(values, index=self._index._data)

    # utilities --------------------------------------------------------
    @property
    def index(self) -> Index:
        return self._index

    @index.setter
    def index(self, values: Iterable[Any]) -> None:
        values = list(values)
        if len(values) != len(self._data):
            raise ValueError("Length mismatch when setting series index")
        self._index = Index(values)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def to_list(self) -> List[Any]:
        return self._data.copy()

    def tolist(self) -> List[Any]:
        return self.to_list()

    def to_numpy(self) -> np.NDArray:
        return np.asarray(self._data)

    values = property(to_numpy)

    def copy(self) -> "Series":
        return Series(self._data, index=self._index._data.copy(), name=self.name)

    def fillna(self, value: float) -> "Series":
        filled = [value if _is_na(v) else v for v in self._data]
        return Series(filled, index=self._index._data)

    def ffill(self) -> "Series":
        filled: List[Any] = []
        last = None
        for value in self._data:
            if _is_na(value):
                filled.append(last if last is not None else value)
            else:
                last = value
                filled.append(value)
        return Series(filled, index=self._index._data)

    def bfill(self) -> "Series":
        filled: List[Any] = [None] * len(self._data)
        next_value = None
        for i in reversed(range(len(self._data))):
            value = self._data[i]
            if _is_na(value):
                filled[i] = next_value if next_value is not None else value
            else:
                next_value = value
                filled[i] = value
        return Series(filled, index=self._index._data)

    def dropna(self) -> "Series":
        values = [v for v in self._data if not _is_na(v)]
        index = [idx for idx, v in zip(self._index._data, self._data) if not _is_na(v)]
        return Series(values, index=index)

    def reindex(self, index: Iterable[Any]) -> "Series":
        new_index = list(index)
        lookup = {label: value for label, value in zip(self._index._data, self._data)}
        values = [lookup.get(label, NA_VALUE) for label in new_index]
        return Series(values, index=new_index)

    def replace(self, to_replace: Iterable[Any] | Any, value: Any) -> "Series":
        if isinstance(to_replace, Iterable) and not isinstance(to_replace, (str, bytes)):
            replacements = set(to_replace)
        else:
            replacements = {to_replace}
        replaced = [value if item in replacements else item for item in self._data]
        return Series(replaced, index=self._index._data)

    def astype(self, dtype) -> "Series":
        if dtype is float:
            return Series([float(v) if not _is_na(v) else NA_VALUE for v in self._data], index=self._index._data)
        if dtype is int:
            return Series([int(v) if not _is_na(v) else 0 for v in self._data], index=self._index._data)
        return self.copy()

    def mean(self) -> float:
        values = [v for v in self._data if not _is_na(v)]
        return float(sum(values) / len(values)) if values else 0.0

    def std(self, ddof: int = 0) -> float:
        values = [v for v in self._data if not _is_na(v)]
        n = len(values)
        if n == 0 or n - ddof <= 0:
            return 0.0
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / (n - ddof)
        return math.sqrt(variance)

    def sum(self) -> float:
        return float(sum(v for v in self._data if not _is_na(v)))

    def abs(self) -> "Series":
        return Series([abs(v) if not _is_na(v) else v for v in self._data], index=self._index._data)

    def max(self) -> float:
        values = [v for v in self._data if not _is_na(v)]
        return max(values) if values else 0.0

    def min(self) -> float:
        values = [v for v in self._data if not _is_na(v)]
        return min(values) if values else 0.0

    def clip(self, lower: float, upper: float) -> "Series":
        return Series([max(min(v, upper), lower) if not _is_na(v) else v for v in self._data], index=self._index._data)

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        values = sorted(v for v in self._data if not _is_na(v))
        if not values:
            return 0.0
        pos = q * (len(values) - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return values[int(pos)]
        fraction = pos - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction

    def cummax(self) -> "Series":
        running: List[float] = []
        current = -float("inf")
        for value in self._data:
            if _is_na(value):
                running.append(current)
            else:
                current = max(current, value)
                running.append(current)
        return Series(running, index=self._index._data)

    def pct_change(self) -> "Series":
        changes: List[float] = [0.0]
        for i in range(1, len(self._data)):
            prev = self._data[i - 1]
            cur = self._data[i]
            if _is_na(prev) or prev == 0:
                changes.append(0.0)
            else:
                changes.append((cur - prev) / prev)
        return Series(changes, index=self._index._data)

    def rolling(self, window: int) -> "RollingSeries":
        return RollingSeries(self, window)

    def between(self, left: float, right: float) -> "Series":
        result = [left <= v <= right if not _is_na(v) else False for v in self._data]
        return Series(result, index=self._index._data)

    def isna(self) -> "Series":
        return Series([_is_na(v) for v in self._data], index=self._index._data)

    def any(self) -> bool:
        return any(bool(v) for v in self._data)

    def all(self) -> bool:
        return all(bool(v) for v in self._data)

    # location ---------------------------------------------------------
    class _Loc:
        def __init__(self, parent: "Series") -> None:
            self._parent = parent

        def __getitem__(self, key):
            return self._parent[key]

        def __setitem__(self, key, value) -> None:
            self._parent[key] = value

    @property
    def loc(self) -> "Series._Loc":
        return Series._Loc(self)


class RollingSeries:
    def __init__(self, series: Series, window: int) -> None:
        self._series = series
        self._window = max(int(window), 1)

    def std(self) -> Series:
        values: List[float] = []
        data = self._series._data
        for i in range(len(data)):
            if i + 1 < self._window:
                values.append(0.0)
                continue
            window_slice = [v for v in data[i + 1 - self._window : i + 1] if not _is_na(v)]
            if not window_slice:
                values.append(0.0)
            else:
                mean_val = sum(window_slice) / len(window_slice)
                variance = sum((v - mean_val) ** 2 for v in window_slice) / len(window_slice)
                values.append(math.sqrt(variance))
        return Series(values, index=self._series._index._data)


# ---------------------------------------------------------------------------
# DataFrame implementation


class DataFrame:
    def __init__(self, data: Mapping[str, Iterable[Any]] | None = None, index: Iterable[Any] | None = None, columns: Iterable[str] | None = None) -> None:
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            if isinstance(data, list):
                columns = list(columns) if columns is not None else sorted({key for row in data for key in row.keys()})
                converted: Dict[str, List[Any]] = {col: [] for col in columns}
                for row in data:
                    if not isinstance(row, Mapping):
                        raise TypeError("DataFrame list entries must be mappings")
                    for col in columns:
                        converted[col].append(Series._coerce_value(row.get(col, NA_VALUE)))
                data = converted
            else:
                raise TypeError("DataFrame expects a mapping of columns")
        keys = list(columns) if columns is not None else list(data.keys())
        series_data: Dict[str, List[Any]] = {}
        for key in keys:
            value = data.get(key, [])
            if isinstance(value, Series):
                raw_values = value.to_list()
            elif hasattr(value, "values"):
                raw_values = value.values.flatten()
            elif isinstance(value, np.NDArray):
                raw_values = value.flatten()
            elif isinstance(value, (int, float, bool, str)):
                raw_values = [value]
            else:
                try:
                    raw_values = list(value)
                except TypeError:
                    raw_values = [value]
            series_data[key] = [Series._coerce_value(v) for v in raw_values]
        lengths = [len(values) for values in series_data.values()]
        expected_length = max(lengths) if lengths else 0
        for key, values in series_data.items():
            if len(values) == expected_length:
                continue
            if len(values) == 1 and expected_length > 1:
                series_data[key] = values * expected_length
            elif len(values) == 0 and expected_length > 0:
                series_data[key] = [NA_VALUE] * expected_length
            else:
                raise ValueError("Column length mismatch")
        if index is None:
            index = list(range(expected_length))
        index = list(index)
        for key in keys:
            column = series_data.setdefault(key, [])
            while len(column) < len(index):
                column.append(NA_VALUE)
        self._index = Index(index)
        self._data = series_data
        self._columns = keys

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key: str | Series) -> Series | "DataFrame":
        if isinstance(key, Series):
            mask = [bool(v) for v in key.to_list()]
            data = {col: [self._data[col][i] for i, flag in enumerate(mask) if flag] for col in self._columns}
            index = [self._index._data[i] for i, flag in enumerate(mask) if flag]
            return DataFrame(data, index=index)
        if key not in self._data:
            raise KeyError(key)
        return Series(self._data[key], index=self._index._data, name=key)

    def __contains__(self, key: str) -> bool:
        return key in self._columns

    def __setitem__(self, key: str, value: Iterable[Any]) -> None:
        values = list(value)
        if len(values) != len(self._index):
            raise ValueError("Length mismatch for column assignment")
        self._data[key] = [Series._coerce_value(v) for v in values]
        if key not in self._columns:
            self._columns.append(key)

    # ------------------------------------------------------------------
    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def index(self) -> Index:
        return self._index

    @index.setter
    def index(self, values: Iterable[Any]) -> None:
        values = list(values)
        if len(values) != len(self._index):
            raise ValueError("Length mismatch when setting DataFrame index")
        self._index = Index(values)

    @property
    def empty(self) -> bool:
        return len(self) == 0 or not self._columns

    @property
    def values(self) -> np.NDArray:
        rows: List[List[float]] = []
        for i in range(len(self._index)):
            row = []
            for col in self._columns:
                row.append(self._data[col][i])
            rows.append(row)
        return np.NDArray(rows)

    def dropna(self, subset: Iterable[str] | None = None) -> "DataFrame":
        columns = list(subset) if subset is not None else self._columns
        keep_indices = []
        for row_idx in range(len(self._index)):
            if any(_is_na(self._data[col][row_idx]) for col in columns):
                continue
            keep_indices.append(row_idx)
        data = {col: [self._data[col][i] for i in keep_indices] for col in self._columns}
        index = [self._index._data[i] for i in keep_indices]
        return DataFrame(data, index=index)

    def reindex(self, index: Iterable[Any]) -> "DataFrame":
        new_index = list(index)
        data: Dict[str, List[Any]] = {col: [] for col in self._columns}
        lookup = {key: pos for pos, key in enumerate(self._index._data)}
        for label in new_index:
            pos = lookup.get(label)
            if pos is None:
                for col in self._columns:
                    data[col].append(NA_VALUE)
            else:
                for col in self._columns:
                    data[col].append(self._data[col][pos])
        return DataFrame(data, index=new_index)

    def ffill(self) -> "DataFrame":
        data: Dict[str, List[Any]] = {}
        for col in self._columns:
            series = Series(self._data[col], index=self._index._data)
            data[col] = series.ffill()._data
        return DataFrame(data, index=self._index._data)

    def bfill(self) -> "DataFrame":
        data: Dict[str, List[Any]] = {}
        for col in self._columns:
            series = Series(self._data[col], index=self._index._data)
            data[col] = series.bfill()._data
        return DataFrame(data, index=self._index._data)

    def fillna(self, value: float) -> "DataFrame":
        data = {
            col: [value if _is_na(v) else v for v in self._data[col]]
            for col in self._columns
        }
        return DataFrame(data, index=self._index._data)

    def copy(self) -> "DataFrame":
        data = {col: values.copy() for col, values in self._data.items()}
        return DataFrame(data, index=self._index._data.copy())

    def sort_index(self, axis: int = 0) -> "DataFrame":
        if axis == 1:
            columns = sorted(self._columns)
            data = {col: self._data[col] for col in columns}
            return DataFrame(data, index=self._index._data)
        order = sorted(range(len(self._index)), key=lambda i: self._index._data[i])
        data = {col: [self._data[col][i] for i in order] for col in self._columns}
        index = [self._index._data[i] for i in order]
        return DataFrame(data, index=index)

    def sort_values(self, column: str, ascending: bool = True) -> "DataFrame":
        if column not in self._columns:
            return self.copy()
        order = sorted(range(len(self._index)), key=lambda i: self._data[column][i], reverse=not ascending)
        data = {col: [self._data[col][i] for i in order] for col in self._columns}
        index = [self._index._data[i] for i in order]
        return DataFrame(data, index=index)

    def abs(self) -> "DataFrame":
        data = {col: [abs(v) if not _is_na(v) else v for v in self._data[col]] for col in self._columns}
        return DataFrame(data, index=self._index._data)

    def sum(self, axis: int = 0) -> Series:
        if axis == 1:
            totals = []
            for i in range(len(self._index)):
                totals.append(sum(self._data[col][i] for col in self._columns if not _is_na(self._data[col][i])))
            return Series(totals, index=self._index._data)
        return Series({col: sum(v for v in self._data[col] if not _is_na(v)) for col in self._columns})

    def max(self, axis: int = 0) -> Series:
        if axis == 1:
            maxima: List[float] = []
            for i in range(len(self._index)):
                row = [self._data[col][i] for col in self._columns if not _is_na(self._data[col][i])]
                maxima.append(max(row) if row else 0.0)
            return Series(maxima, index=self._index._data)
        return Series({col: max(v for v in self._data[col] if not _is_na(v)) for col in self._columns})

    def astype(self, dtype) -> "DataFrame":
        data = {col: Series(self._data[col], index=self._index._data).astype(dtype)._data for col in self._columns}
        return DataFrame(data, index=self._index._data)

    def to_csv(self, handle, index: bool = True) -> None:
        writer = csv.writer(handle)
        header = (["index"] if index else []) + self._columns
        writer.writerow(header)
        for pos, label in enumerate(self._index._data):
            row = [self._data[col][pos] for col in self._columns]
            if index:
                row = [label] + row
            writer.writerow(row)

    def to_parquet(self, path: Path) -> None:
        raise ImportError("parquet support not available in lightweight pandas shim")

    # loc ----------------------------------------------------------------
    class _Loc:
        def __init__(self, frame: "DataFrame") -> None:
            self._frame = frame

        def __getitem__(self, key):
            if isinstance(key, tuple):
                rows = self._select_rows(key[0])
                return self._select_columns(rows, key[1])
            return self._select_rows(key)

        def _select_rows(self, row_key):
            if isinstance(row_key, (DatetimeIndex, Index)):
                labels = row_key.to_list()
            elif isinstance(row_key, Series):
                mask = [bool(v) for v in row_key.to_list()]
                labels = [label for label, flag in zip(self._frame._index._data, mask) if flag]
            elif isinstance(row_key, list) and row_key and isinstance(row_key[0], bool):
                labels = [label for label, flag in zip(self._frame._index._data, row_key) if flag]
            elif isinstance(row_key, list):
                labels = row_key
            else:
                labels = [row_key]

            if len(labels) == 1:
                label = labels[0]
                if label not in self._frame._index:
                    raise KeyError(label)
                pos = self._frame._index._data.index(label)
                row = {col: self._frame._data[col][pos] for col in self._frame._columns}
                return Series(row, index=self._frame._columns)

            data = {col: [] for col in self._frame._columns}
            new_index = []
            for label in labels:
                if label not in self._frame._index:
                    raise KeyError(label)
                pos = self._frame._index._data.index(label)
                for col in self._frame._columns:
                    data[col].append(self._frame._data[col][pos])
                new_index.append(label)
            return DataFrame(data, index=new_index)

        def _select_columns(self, obj, col_key):
            if isinstance(obj, Series):
                if isinstance(col_key, list):
                    return Series([obj[col] for col in col_key], index=col_key)
                return obj[col_key]
            if isinstance(col_key, (list, tuple)):
                data = {col: obj._data[col] for col in col_key}
                return DataFrame(data, index=obj._index._data)
            return obj[col_key]

        def __setitem__(self, key, value) -> None:
            if key not in self._frame._index:
                raise KeyError(key)
            pos = self._frame._index._data.index(key)
            if isinstance(value, Series):
                row_values = value
            else:
                row_values = Series(value, index=self._frame._columns)
            for col in self._frame._columns:
                self._frame._data[col][pos] = row_values[col]

    @property
    def loc(self) -> "DataFrame._Loc":
        return DataFrame._Loc(self)

    class _ILoc:
        def __init__(self, frame: "DataFrame") -> None:
            self._frame = frame

        def __getitem__(self, key):
            if isinstance(key, int):
                pos = key if key >= 0 else len(self._frame._index) + key
                if pos < 0 or pos >= len(self._frame._index):
                    raise IndexError("iloc index out of range")
                row = {col: self._frame._data[col][pos] for col in self._frame._columns}
                return Series(row, index=self._frame._columns)
            if isinstance(key, slice):
                indices = list(range(*key.indices(len(self._frame._index))))
                data = {col: [self._frame._data[col][i] for i in indices] for col in self._frame._columns}
                index = [self._frame._index._data[i] for i in indices]
                return DataFrame(data, index=index)
            raise TypeError("Unsupported iloc indexer")

    @property
    def iloc(self) -> "DataFrame._ILoc":
        return DataFrame._ILoc(self)


# ---------------------------------------------------------------------------
# Module level helpers


def Series_from_array(values: Iterable[Any], index: Iterable[Any]) -> Series:  # pragma: no cover - convenience
    return Series(values, index=index)


def concat(frames: Iterable[DataFrame], ignore_index: bool = False) -> DataFrame:
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return DataFrame()
    columns = frames[0].columns
    data: Dict[str, List[Any]] = {col: [] for col in columns}
    index: List[Any] = []
    for frame in frames:
        for col in columns:
            data[col].extend(frame._data[col])
        if ignore_index:
            index.extend(range(len(index), len(index) + len(frame.index)))
        else:
            index.extend(frame.index.to_list())
    return DataFrame(data, index=index)


def date_range(start: Any, periods: int, freq: str = "D") -> DatetimeIndex:
    start_dt = _ensure_datetime(start)
    items: List[datetime] = []
    current = start_dt
    freq = freq or "D"
    freq_norm = freq.lower()
    if freq_norm in {"b", "d", "1d", "1 day"}:
        step = timedelta(days=1)
    elif freq_norm in {"h", "1h", "1 hour"}:
        step = timedelta(hours=1)
    elif freq_norm in {"t", "1t", "1min", "1 minute", "min"}:
        step = timedelta(minutes=1)
    else:
        step = timedelta(days=1)
    while len(items) < periods:
        if freq_norm.startswith("b") and current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        items.append(current)
        current += step
    return DatetimeIndex(items)


def TimedeltaIndex(*args, **kwargs):  # pragma: no cover - unused helper
    raise NotImplementedError


def read_csv(path: str | Path, index_col: int | None = None, parse_dates: Iterable[int] | None = None) -> DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    header = rows[0]
    body = rows[1:]
    columns = header[1:] if index_col == 0 else header
    data: Dict[str, List[Any]] = {col: [] for col in columns}
    index: List[Any] = []
    for row in body:
        if index_col is not None:
            idx_value = row[index_col]
            if parse_dates and index_col in parse_dates:
                idx_value = _ensure_datetime(idx_value)
            index.append(idx_value)
            values = row[:index_col] + row[index_col + 1 :]
        else:
            index.append(len(index))
            values = row
        for col, value in zip(columns, values):
            try:
                data[col].append(float(value))
            except ValueError:
                data[col].append(value)
    return DataFrame(data, index=index)


def to_datetime(values: Iterable[Any]) -> DatetimeIndex:  # pragma: no cover - small helper
    return DatetimeIndex(values)


def DataFrame_from_records(records: Iterable[Mapping[str, Any]]) -> DataFrame:  # pragma: no cover - helper
    data: Dict[str, List[Any]] = {}
    index: List[int] = []
    for i, record in enumerate(records):
        for key, value in record.items():
            data.setdefault(key, []).append(value)
        index.append(i)
    return DataFrame(data, index=index)


def Series_from_dict(data: Mapping[Any, Any]) -> Series:  # pragma: no cover - helper
    return Series(data)


__all__ = [
    "Index",
    "DatetimeIndex",
    "Timedelta",
    "Timestamp",
    "Series",
    "DataFrame",
    "concat",
    "date_range",
    "read_csv",
]
