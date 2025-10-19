"""A tiny subset of NumPy used for the kata test-suite.

The real project depends heavily on :mod:`numpy` but the execution
environment available to the kata intentionally avoids third party
dependencies.  The tests only exercise a sliver of the full NumPy API so
this module provides just enough functionality for the rest of the
codebase to run.  The implementation favours clarity over raw
performance and only implements the pieces that are actually needed in
the tests (vector arithmetic, a couple of aggregation helpers and some
array creation utilities).

The provided :class:`NDArray` type mimics a minimal 1D/2D ndarray.  It is
backed by Python lists and implements the handful of operations the code
under test requires (basic indexing, element wise arithmetic and a
couple of statistics helpers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, Union
import builtins

Number = Union[int, float]


class generic(float):  # pragma: no cover - compatibility shim
    pass


def _as_list(value: Union["NDArray", Sequence[Number]]) -> List[float]:
    if isinstance(value, NDArray):
        return value.flatten()
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v) for v in value]


def _apply_binary(
    lhs: "NDArray | Sequence[Number]",
    rhs: "NDArray | Sequence[Number] | Number",
    op,
) -> "NDArray":
    left = NDArray(lhs)
    if isinstance(rhs, (int, float)):
        return NDArray([[op(v, rhs) for v in row] for row in left._data])
    right = NDArray(rhs)
    if left.ndim != right.ndim:
        raise ValueError("Operand dimensions do not match")
    if left.ndim == 1:
        lvals = left.flatten()
        rvals = right.flatten()
        if len(lvals) != len(rvals):
            raise ValueError("Operand lengths differ")
        return NDArray([op(a, b) for a, b in zip(lvals, rvals)])
    if left.shape != right.shape:
        raise ValueError("Operand shapes differ")
    return NDArray([[op(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(left._data, right._data)])


@dataclass
class NDArray:
    """Minimal ndarray wrapper with very small API surface."""

    _data: List[List[float]]

    def __init__(
        self,
        values: Union[
            "NDArray",
            Sequence[Sequence[Number]] | Sequence[Number],
            Number,
        ],
    ) -> None:
        if isinstance(values, NDArray):
            self._data = [row.copy() for row in values._data]
            return
        if isinstance(values, (int, float)):
            self._data = [[float(values)]]
            return
        if not isinstance(values, Sequence):
            raise TypeError("Unsupported type for NDArray")
        if values and isinstance(values[0], Sequence):
            self._data = [[float(v) for v in row] for row in values]  # type: ignore[arg-type]
        else:
            self._data = [[float(v) for v in values]]  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[float] | Iterator[List[float]]:
        if self.ndim == 1:
            return iter(self._data[0])
        return iter(self._data)

    def __len__(self) -> int:
        if self.ndim == 1:
            return len(self._data[0])
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            row_subset = self._slice_rows(rows)
            return self._slice_cols(row_subset, cols)
        if isinstance(key, slice):
            return NDArray(self._data[key])
        if isinstance(key, int):
            row = self._data[key]
            return NDArray([row]) if self.ndim > 1 else row[0]
        raise TypeError("Unsupported index type for NDArray")

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            rows, cols = key
            row_subset = self._slice_rows(rows)
            values = NDArray(value)
            if values.ndim == 1:
                values = NDArray([values._data[0] for _ in range(len(row_subset))])
            if len(row_subset) != len(values._data):
                raise ValueError("Row count mismatch during assignment")
            for row_idx, target_row in enumerate(row_subset):
                replacement = values._data[row_idx]
                if isinstance(cols, int):
                    target_row[cols] = replacement[0]
                elif isinstance(cols, slice):
                    target_row[cols] = replacement
                else:
                    for c_idx, col in enumerate(cols):
                        target_row[col] = replacement[c_idx]
            return
        if isinstance(key, int):
            seq = NDArray(value)
            self._data[key] = seq._data[0]
            return
        raise TypeError("Unsupported index assignment for NDArray")

    # ------------------------------------------------------------------
    def _slice_rows(self, rows) -> List[List[float]]:
        if isinstance(rows, slice):
            return self._data[rows]
        if isinstance(rows, (list, tuple)):
            return [self._data[i] for i in rows]
        if isinstance(rows, int):
            return [self._data[rows]]
        return [self._data[idx] for idx, flag in enumerate(rows) if flag]

    def _slice_cols(self, rows: List[List[float]], cols):
        if isinstance(cols, slice):
            return NDArray([row[cols] for row in rows])
        if isinstance(cols, (list, tuple)):
            return NDArray([[row[c] for c in cols] for row in rows])
        if isinstance(cols, int):
            return NDArray([row[cols] for row in rows])
        return NDArray([[row[idx] for idx, flag in enumerate(cols) if flag] for row in rows])

    # ------------------------------------------------------------------
    def to_list(self) -> List[List[float]]:
        return [row.copy() for row in self._data]

    def flatten(self) -> List[float]:
        if self.ndim == 1:
            return self._data[0].copy()
        flattened: List[float] = []
        for row in self._data:
            flattened.extend(row)
        return flattened

    @property
    def ndim(self) -> int:
        return 1 if len(self._data) <= 1 else 2

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.ndim == 1:
            return (len(self._data[0]),)
        cols = len(self._data[0]) if self._data else 0
        return (len(self._data), cols)

    @property
    def size(self) -> int:
        if self.ndim == 1:
            return len(self._data[0])
        if not self._data:
            return 0
        return len(self._data) * len(self._data[0])

    def sum(self) -> float:
        return float(sum(self.flatten()))

    def mean(self) -> float:
        flat = self.flatten()
        return float(sum(flat) / len(flat)) if flat else 0.0

    def astype(self, dtype) -> "NDArray":
        if dtype in (int, float):
            return NDArray([[dtype(v) for v in row] for row in self._data])
        return NDArray(self)

    def clip(self, min_value: Number, max_value: Number) -> "NDArray":
        return NDArray([[max(min(v, max_value), min_value) for v in row] for row in self._data])

    def cumprod(self) -> "NDArray":
        flat = self.flatten()
        result: List[float] = []
        prod = 1.0
        for value in flat:
            prod *= value
            result.append(prod)
        return NDArray(result)

    # ------------------------------------------------------------------
    def _binary_op(self, other, op) -> "NDArray":
        return _apply_binary(self, other, op)

    def __add__(self, other) -> "NDArray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "NDArray":
        if isinstance(other, (int, float)):
            return self.__add__(other)
        return NDArray(other).__add__(self)

    def __sub__(self, other) -> "NDArray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "NDArray":
        if isinstance(other, (int, float)):
            return NDArray([other - v for v in self.flatten()])
        return NDArray(other).__sub__(self)

    def __mul__(self, other) -> "NDArray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "NDArray":
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NDArray(other).__mul__(self)

    def __truediv__(self, other) -> "NDArray":
        return self._binary_op(other, lambda a, b: a / b if b != 0 else 0.0)

    def __rtruediv__(self, other) -> "NDArray":
        if isinstance(other, (int, float)):
            return NDArray(other).__truediv__(self)
        return NDArray(other).__truediv__(self)

    def __pow__(self, power) -> "NDArray":
        return NDArray([[v**power for v in row] for row in self._data])

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"NDArray({self._data!r})"


# ---------------------------------------------------------------------------
# Array creation helpers


def array(values: Sequence[Number] | Sequence[Sequence[Number]]) -> NDArray:
    return NDArray(values)


def asarray(values: Sequence[Number] | Sequence[Sequence[Number]] | NDArray, dtype=float) -> NDArray:
    arr = NDArray(values)
    if dtype in (int, float):
        return arr.astype(dtype)
    return arr


def linspace(start: float, stop: float, num: int) -> NDArray:
    if num <= 1:
        return NDArray([start])
    step = (stop - start) / (num - 1)
    return NDArray([start + step * i for i in range(num)])


def sin(values: Union[NDArray, Sequence[Number], Number]) -> NDArray:
    arr = NDArray(values)
    return NDArray([[math.sin(v) for v in row] for row in arr._data])


def cos(values: Union[NDArray, Sequence[Number], Number]) -> NDArray:
    arr = NDArray(values)
    return NDArray([[math.cos(v) for v in row] for row in arr._data])


def sqrt(values: Union[NDArray, Sequence[Number], Number]) -> Union[NDArray, float]:
    if isinstance(values, (int, float)):
        return math.sqrt(values)
    arr = NDArray(values)
    return NDArray([[math.sqrt(v) for v in row] for row in arr._data])


def mean(values: Union[NDArray, Sequence[Number]]) -> float:
    return NDArray(values).mean()


def diff(values: Union[NDArray, Sequence[Number]], prepend: Number | None = None) -> NDArray:
    arr = NDArray(values)
    data = arr.flatten()
    if prepend is not None:
        data = [float(prepend)] + data
    diffs = [data[i] - data[i - 1] for i in range(1, len(data))]
    return NDArray(diffs)


def abs(values: Union[NDArray, Sequence[Number]]) -> NDArray:
    arr = NDArray(values)
    return NDArray([[builtins.abs(v) for v in row] for row in arr._data])


def column_stack(arrays: Sequence[Union[NDArray, Sequence[Number]]]) -> NDArray:
    columns = [NDArray(a).flatten() for a in arrays]
    if not columns:
        return NDArray([])
    rows = zip(*columns)
    return NDArray([list(row) for row in rows])


def full(length: int, fill_value: Number) -> NDArray:
    return NDArray([fill_value] * int(length))


def zeros(length: int) -> NDArray:
    return full(length, 0.0)


def clip(values: Union[NDArray, Sequence[Number], Number], a_min: Number, a_max: Number) -> NDArray:
    if isinstance(values, (int, float)):
        return float(max(min(values, a_max), a_min))
    arr = NDArray(values)
    return arr.clip(a_min, a_max)


def allclose(
    a: Union[NDArray, Sequence[Number]],
    b: Union[NDArray, Sequence[Number]],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    arr_a = NDArray(a).flatten()
    arr_b = NDArray(b).flatten()
    if len(arr_a) != len(arr_b):
        return False
    for lhs, rhs in zip(arr_a, arr_b):
        if builtins.abs(lhs - rhs) > (atol + rtol * builtins.abs(rhs)):
            return False
    return True


def unique(values: Union[NDArray, Sequence[Number]]) -> NDArray:
    arr = NDArray(values).flatten()
    return NDArray(sorted(set(arr)))


def nanstd(values: Union[NDArray, Sequence[Number]]) -> float:
    arr = [v for v in NDArray(values).flatten() if not math.isnan(v)]
    if not arr:
        return 0.0
    mean_val = sum(arr) / len(arr)
    variance = sum((v - mean_val) ** 2 for v in arr) / len(arr)
    return math.sqrt(variance)


def isnan(value: Number) -> bool:
    return math.isnan(value)


pi = math.pi
nan = float("nan")
inf = float("inf")


__all__ = [
    "NDArray",
    "ndarray",
    "generic",
    "array",
    "asarray",
    "linspace",
    "sin",
    "cos",
    "sqrt",
    "mean",
    "diff",
    "abs",
    "column_stack",
    "full",
    "zeros",
    "clip",
    "allclose",
    "unique",
    "nanstd",
    "isnan",
    "pi",
    "nan",
    "inf",
]


ndarray = NDArray
