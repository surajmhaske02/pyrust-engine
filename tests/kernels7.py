"""Kernels for 0.7.0 features."""
import math
import numpy as np
from pyrust_engine import F64Array, F64Array2, I64Array, prange


def any_negative(values: F64Array) -> bool:
    found = False
    for x in values:
        if x < 0:
            found = True
            break
    return found


def count_flags(n: int, flag: bool) -> int:
    total = 0
    for i in range(n):
        even = i % 2 == 0
        if even and flag:
            total += 1
        total += even          # bool used as int
        total += flag & even   # bool & bool is a bool
    return total


def iterate(values: F64Array) -> float:
    total = 0.0
    for x in values:
        total += x * x
    return total


def enumerate_weighted(values: F64Array) -> float:
    total = 0.0
    for i, x in enumerate(values, 1):
        total += i * x
    return total


def builtins_mix(values: F64Array, a: int, b: int) -> float:
    q, r = divmod(a, b)
    fq, fr = divmod(float(a), 2.5)
    total = sum(values) + q + r + fq + fr + round(2.5) + round(3.5) + round(-0.5)
    total += math.atan2(1.0, -1.0) + math.copysign(3.0, -0.0) + math.fmod(7.5, 2.0)
    return total


def stats(values: F64Array):
    n = len(values)
    total = 0.0
    for x in values:
        total += x
    mean = total / n
    var = 0.0
    for x in values:
        var += (x - mean) ** 2
    return mean, var / n


def fib_pair(n: int):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a, b


def scaled(x: float, factor: float = 2.0, offset: float = 0.5) -> float:
    return x * factor + offset


def uses_defaults(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += scaled(float(i)) + scaled(float(i), offset=1.0) + scaled(x=1.0, factor=3.0)
    return total


def untyped_sum(n, step):
    # no annotations: compiled for whatever types are passed (int or float)
    total = 0
    for i in range(n):
        total += i * step
    return total


def matrix_trace(m: F64Array2) -> float:
    rows, cols = m.shape
    total = 0.0
    for i in range(min(rows, cols)):
        total += m[i, i]
    return total


def matrix_scale(m: F64Array2, factor: float):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i, j] *= factor


def outer(a: F64Array, b: F64Array):
    out = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]
    return out


def cumulative(values: F64Array):
    out = np.empty_like(values)
    total = 0.0
    for i, x in enumerate(values):
        total += x
        out[i] = total
    return out


def int_histogram(values: I64Array, bins: int):
    counts = np.zeros(bins, dtype=np.int64)
    for v in values:
        counts[v % bins] += 1
    return counts


def write_many(values: F64Array, n: int) -> int:
    for k in range(n):
        values[k % len(values)] += 1.0
    return 10 // (n - 1000)  # fails when n == 1000, after many writes


def parallel_square(values: F64Array, out: F64Array) -> int:
    count = 0
    for i in prange(len(values)):
        x = values[i]
        out[i] = x * x + math.sqrt(abs(x))
        if x > 0:
            count += 1
    return count


def parallel_fail(values: F64Array, out: F64Array):
    for i in prange(len(values)):
        out[i] = 1.0 / values[i]   # ZeroDivisionError on a zero


def long_loop(n: int) -> int:
    total = 0
    for i in range(n):
        total += i % 7
    return total


def bad_parallel_float_sum(values: F64Array) -> float:
    total = 0.0
    for i in prange(len(values)):
        total += values[i]
    return total


def bad_parallel_neighbour(values: F64Array, out: F64Array):
    for i in prange(1, len(values)):
        out[i] = out[i - 1] + values[i]
