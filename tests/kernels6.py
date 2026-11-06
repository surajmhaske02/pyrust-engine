"""Kernels for 0.6.0: recursion, array writes, int64 arrays, None returns, int/float compares."""
from pyrust_engine import F64Array, I64Array


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def is_even(n: int) -> int:
    return 1 if n == 0 else is_odd(n - 1)


def is_odd(n: int) -> int:
    return 0 if n == 0 else is_even(n - 1)


def depth(n: int) -> int:
    if n == 0:
        return 0
    return depth(n - 1) + 1


def power_sum(n: int, x: float) -> float:
    # recursion with a float result
    if n == 0:
        return 0.0
    return x ** n + power_sum(n - 1, x)


def scale(values: F64Array, out: F64Array, factor: float):
    for i in range(len(values)):
        out[i] = float(values[i]) * factor


def prefix_sum(values: F64Array, out: F64Array) -> float:
    total = 0.0
    for i in range(len(values)):
        total += float(values[i])
        out[i] = total
    return total


def normalize(values: F64Array) -> None:
    total = 0.0
    for i in range(len(values)):
        total += float(values[i])
    for i in range(len(values)):
        values[i] /= total


def accumulate(values: F64Array, n: int):
    for i in range(n):
        values[i % len(values)] += 1
        values[-1] -= 0.5
        values[0] *= 1.0


def histogram(data: I64Array, counts: I64Array, bins: int) -> None:
    for i in range(len(data)):
        counts[int(data[i]) % bins] += 1


def int_sum(data: I64Array) -> int:
    total = 0
    for i in range(len(data)):
        total += int(data[i])
    return total


def int_to_float(data: I64Array, out: F64Array):
    for i in range(len(data)):
        out[i] = float(data[i]) / 2.0


def write_then_fail(out: F64Array, n: int) -> int:
    total = 0
    for i in range(n):
        out[i] = 1.0
        total += 10 // (i - 5)   # ZeroDivisionError when i == 5
    return total


def copy_shift(src: F64Array, dst: F64Array):
    for i in range(1, len(src)):
        dst[i] = float(src[i - 1])


def count_below(limit: float, n: int) -> int:
    count = 0
    for i in range(n):
        if i < limit:
            count += 1
        if limit >= i:
            count += 10
        if i == limit:
            count += 100
    return count


def compare_pairs(i: int, f: float) -> int:
    result = 0
    for _ in range(1):
        if i < f:
            result += 1
        if i <= f:
            result += 2
        if i > f:
            result += 4
        if i >= f:
            result += 8
        if i == f:
            result += 16
        if i != f:
            result += 32
        if f < i:
            result += 64
        if f >= i:
            result += 128
    return result
