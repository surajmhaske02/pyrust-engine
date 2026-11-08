"""Functions and classes under test for 0.7.5 (they need real source files to compile)."""
from dataclasses import dataclass

from pyrust_engine import optimize


# ------------------------------------------------------------------- classes
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    steps: int = 0

    def advance(self, dt: float, n: int) -> None:
        for _ in range(n):
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.y < 0.0:
                self.vy = -self.vy
                self.y = -self.y
            self.steps += 1

    def energy(self, mass: float) -> float:
        total = 0.0
        for _ in range(1):
            total += 0.5 * mass * (self.vx * self.vx + self.vy * self.vy)
        return total

    def calls_method(self) -> float:
        return self.energy(1.0)


class Account:
    rate = 0.01  # a class attribute

    def __init__(self, balance, owner):
        self.balance = balance
        self.owner = owner
        self.active = True

    def grow(self, years: int) -> float:
        for _ in range(years):
            if self.active:
                self.balance += self.balance * self.rate
        return self.balance

    def label_length(self) -> int:
        n = 0
        for ch in self.owner:
            if ch != ' ':
                n += 1
        return n


class Slotted:
    __slots__ = ('count', 'total')

    def __init__(self):
        self.count = 0
        self.total = 0

    def add_up(self, n: int) -> int:
        for i in range(n):
            self.count += 1
            self.total += i * i
        return self.total


class WithProperty:
    def __init__(self):
        self._v = 2.0

    @property
    def v(self):
        return self._v * 10

    def triple(self, n: int) -> float:
        t = 0.0
        for _ in range(n):
            t += self.v
        return t


@dataclass(frozen=True)
class Frozen:
    n: int

    def bump(self, k: int) -> int:
        for _ in range(k):
            self.n += 1
        return self.n


class Grid:
    def __init__(self, data):
        self.data = data
        self.scale = 2.0

    def double_all(self) -> float:
        total = 0.0
        for i in range(len(self.data)):
            self.data[i] = self.data[i] * self.scale
            total += self.data[i]
        return total


class Counter:
    def __init__(self, n):
        self.n = n

    def blow_up(self, k: int) -> int:
        for _ in range(k):
            self.n = self.n * 1000
        return self.n

    def to_float(self) -> float:
        for _ in range(3):
            self.n = self.n + 0.5
        return self.n


def transfer(a, b, amount: float) -> None:
    for _ in range(3):
        a.balance -= amount
        b.balance += amount


class Decorated:
    """@optimize used directly on a method in the class body."""

    def __init__(self):
        self.hits = 0

    @optimize(warn_on_fallback=False)
    def hit(self, n: int) -> int:
        for _ in range(n):
            self.hits += 2
        return self.hits


# ------------------------------------------------------------- direct calls
def total(a) -> float:
    t = 0.0
    for x in a:
        t += x
    return t


def int_total(a) -> int:
    t = 0
    for i in range(len(a)):
        t += a[i]
    return t


def scaled_into(a, out, k: float) -> float:
    t = 0.0
    for i in range(len(a)):
        out[i] = a[i] * k
        t += out[i]
    return t


def fill_until(out, stop: int) -> None:
    for i in range(len(out)):
        out[i] = 1.0 / (stop - i)  # ZeroDivisionError at i == stop: earlier writes are undone


def grid_sum(m) -> float:
    t = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            t += m[i, j] * (i + 1)
    return t


def vowels(text: str) -> int:
    n = 0
    for ch in text:
        if ch in 'aeiou':
            n += 1
    return n


def starts_with(text: str, prefix: str) -> bool:
    return text.startswith(prefix) and len(text) > 3


def checksum(data: bytes) -> int:
    h = 0
    for b in data:
        h = (h * 31 + b) % 1000000007
    return h


def mixed(a, text: str, k: int, w: float) -> float:
    t = 0.0
    for i in range(min(len(a), k)):
        t += a[i] * w
    return t + len(text)
