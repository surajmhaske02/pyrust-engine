"""Functions and classes under test for 0.7.7 (they need real source files to compile)."""
import math
from dataclasses import dataclass, field

import numpy as np

from pyrust_engine import optimize


# ------------------------------------------------------- methods calling methods
@dataclass
class Body:
    x: float
    vx: float
    hits: int = 0
    trail: list = field(default_factory=list)

    def bounce(self):
        if self.x < 0.0:
            self.x = -self.x
            self.vx = -self.vx
            self.hits += 1

    def speed(self) -> float:
        return abs(self.vx)

    def advance(self, dt: float, n: int) -> float:
        total = 0.0
        for _ in range(n):
            self.x += self.vx * dt
            self.bounce()
            total += self.speed()
        return total


class Fast(Body):
    def speed(self) -> float:  # overridden: compiled separately for this class
        return 2.0 * abs(self.vx)


class Math:
    def __init__(self):
        self.base = 3

    def fact(self, n: int) -> int:
        if n < 2:
            return self.base
        return n * self.fact(n - 1)

    def total(self, n: int) -> int:
        t = 0
        for i in range(n):
            t += self.fact(i % 10)
        return t

    @staticmethod
    def helper(x):
        return x + 1

    def uses_static(self, n: int) -> int:
        t = 0
        for i in range(n):
            t += self.helper(i)
        return t


class Store:
    def __init__(self, values, weights, names):
        self.values = values  # list of floats: read and written
        self.weights = weights  # dict str -> float: read only
        self.names = names  # list of str: read only

    def scale(self, k: float) -> float:
        t = 0.0
        for i in range(len(self.values)):
            self.values[i] = self.values[i] * k
            t += self.values[i]
        return t

    def score(self) -> float:
        t = 0.0
        for name in self.names:
            t += self.weights.get(name, 0.0)
        return t


# -------------------------------------------------------------- exceptions
def safe_mean(values, n: int) -> float:
    total = 0.0
    for i in range(n):
        try:
            total += values[i]
        except IndexError:
            total += 0.0
    try:
        return total / n
    except ZeroDivisionError:
        return 0.0


def checked_sum(n):
    if n < 0:
        raise ValueError('n must not be negative')
    assert n < 10**7, 'n is too big'
    t = 0
    for i in range(n):
        t += i
    return t


def with_finally(n):
    t = 0
    for i in range(n):
        try:
            t += i
        finally:
            t += 1
    return t


def parse_all(texts: list[str]) -> int:
    total = 0
    for s in texts:
        try:
            total += int(s)
        except ValueError:
            total -= 1
    return total


# ------------------------------------------------------------------- NumPy
def window_sums(a, w: int) -> float:
    best = 0.0
    for i in range(len(a) - w + 1):
        segment = a[i : i + w]
        s = sum(segment)
        if s > best:
            best = s
    return best


def row_norms(m) -> float:
    total = 0.0
    for i in range(m.shape[0]):
        row = m[i, :]
        s = 0.0
        for x in row:
            s += x * x
        total += np.sqrt(s)
    return total


def tail_mean(a) -> float:
    tail = a[-5:]
    t = 0.0
    for x in tail:
        t += x
    return t / len(tail)


def np_math(x: float) -> float:
    return np.sqrt(abs(x)) + np.floor(x) + np.abs(x) + np.square(x) + np.pi + float(np.isnan(x))


def np_math_loop(n: int) -> float:
    t = 0.0
    for i in range(n):
        t += np.sqrt(float(i)) + np.floor(i / 3)
    return t


def threshold(img, out, level: int) -> int:
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > level:
                out[i, j] = 255
                count += 1
            else:
                out[i, j] = 0
    return count


def masked_total(values, mask) -> float:
    t = 0.0
    for i in range(len(values)):
        if mask[i]:
            t += values[i]
    return t


def int32_sum(a) -> int:
    t = 0
    for i in range(len(a)):
        t += int(a[i])
    return t


def bump(a, k: int) -> None:
    for i in range(len(a)):
        a[i] = int(a[i]) + k  # out of range for uint8: NumPy raises OverflowError


def small_math(a) -> int:
    return a[0] + 1  # NumPy uint8 arithmetic wraps: runs as Python


def sqrt_negative(x: float) -> float:
    t = 0.0
    for _ in range(2):
        t += np.sqrt(x)  # NaN and a RuntimeWarning in NumPy: replays in Python
    return t


# ------------------------------------------------------------- direct calls
def list_total(values: list[float]) -> float:
    t = 0.0
    for x in values:
        t += x
    return t


def double_list(values: list[int]) -> None:
    for i in range(len(values)):
        values[i] *= 2


def squares(n: int) -> list[float]:
    out = []
    for i in range(n):
        out.append(math.sqrt(i))
    return out


def counts(n: int) -> list[int]:
    out = []
    for i in range(n):
        out.append(i * i)
    return out


def shout(text: str, times: int) -> str:
    out = ''
    for _ in range(times):
        out += text.upper() + '!'
    return out
