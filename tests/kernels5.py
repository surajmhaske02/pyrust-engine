"""Kernels for 0.5.0 features. Every function here contains a loop, so it compiles natively."""
import math
from math import sqrt as square_root
from pyrust_engine import F64Array, optimize

SCALE = 3
OFFSET = 0.25


def int_ops(n: int) -> int:
    total = 0
    for i in range(n):
        total += (i ** 2) % 1000 + (i & 7) + (i | 3) - (i ^ 5) + (i << 2) + (i >> 1) + ~i
        total += abs(i - 50) + max(i, 10, 3) - min(i, 7)
    return total


def float_ops(n: int, x: float) -> float:
    total = 0.0
    for i in range(1, n):
        total += x ** 0.5 + (x * i) // 3.0 + (x * i) % 2.5 + abs(x - i) + max(x, 1.5) - min(x, -2.0)
        total += i / 3          # int / int gives a float
        total += float(i) ** 2
    return total


def math_ops(n: int, x: float) -> float:
    total = 0.0
    for i in range(1, n):
        y = x + i
        total += math.sqrt(y) + math.exp(-y) + math.log(y) + math.sin(y) + math.cos(y)
        total += math.atan(y) + math.fabs(-y) + math.log10(y) + math.log2(y) + math.tanh(y)
        total += math.floor(y) + math.ceil(y) + int(y) + square_root(y) + math.pi * math.e
    return total


def math_domain(x: float) -> float:
    total = 0.0
    for i in range(3):
        total += math.sqrt(x - i)
    return total


def float_pow(a: float, b: float) -> float:
    result = 0.0
    for i in range(1):
        result = a ** b
    return result


def int_pow(a: int, b: int) -> int:
    result = 0
    for i in range(1):
        result = a ** b
    return result


def ternary_and_while_true(n: int) -> int:
    count = 0
    i = 0
    while True:
        i += 1
        count += 2 if i % 3 == 0 else 1
        if i >= n:
            break
    return count


def globals_used(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += i * SCALE + OFFSET
    return total


def nan_checks(x: float) -> int:
    count = 0
    for i in range(1):
        if math.isnan(x):
            count += 1
        if math.isinf(x):
            count += 10
        if math.isfinite(x):
            count += 100
    return count


def helper_square(v: float) -> float:
    return v * v


def helper_sum(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total


@optimize
def decorated_helper(a: int, b: int) -> int:
    return a * b + 1


def calls_helpers(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += helper_square(float(i)) + helper_sum(i % 10) + decorated_helper(i, 2)
    return total


def array_helper(values: F64Array, i: int) -> float:
    return float(values[i]) * 2.0


def calls_with_array(values: F64Array) -> float:
    total = 0.0
    for i in range(len(values)):
        total += array_helper(values, i)
    return total


def recursive(n: int) -> int:
    total = 0
    for i in range(1):
        total = n if n < 2 else recursive(n - 1) + recursive(n - 2)
    return total


def shift_edges(a: int, b: int) -> int:
    result = 0
    for i in range(1):
        result = (a << b) + (a >> b)
    return result


def int_true_division(a: int, b: int) -> float:
    result = 0.0
    for i in range(1):
        result = a / b
    return result


def float_floor_mod(a: float, b: float) -> float:
    result = 0.0
    for i in range(1):
        result = (a // b) * 1000.0 + a % b
    return result


def to_int(x: float) -> int:
    result = 0
    for i in range(1):
        result = int(x)
    return result


def wraps_helper(n: int) -> int:
    # No loop of its own, but it calls a looping function: worth compiling together.
    return helper_sum(n) + 1


def recursive_unannotated(n: int):
    total = 0
    for i in range(1):
        total = n if n < 2 else recursive_unannotated(n - 1) + recursive_unannotated(n - 2)
    return total
