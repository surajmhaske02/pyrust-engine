"""Functions under test for 0.7.4 (they need real source files to compile)."""
import math

from pyrust_engine import optimize


def add_ints(a, b):
    return a + b


def mix(a: float, b: int, flag: bool) -> float:
    return a * b if flag else a - b


def is_close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def nothing(n):
    for i in range(n):
        pass


def with_default(a, b=10):
    return a + b


def scale(x):
    return helper(x) + 1


def helper(x):
    return x * 2


def root(x: float) -> float:
    return math.sqrt(x)


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def spin(n):
    t = 0
    for i in range(n):
        t += i & 7
    return t


def grow(n):
    t = 1
    for i in range(n):
        t = t * 3
    return t


def divide(a, b):
    return a // b


@optimize(native_loop_free=True, warn_on_fallback=False)
def picklable(a, b):
    return a - b


class Holder:
    """An optimized function stored on a class still binds like a method."""

    @optimize(native_loop_free=True, warn_on_fallback=False)
    def method(self, n):
        return n + 1


# ---------------------------------------------------------------- string views
def middle(text: str) -> str:
    return text[3:-3]


def grown(text: str) -> str:
    part = text[2:40]
    part += '!'
    return part


def pieces(text: str) -> dict[str, int]:
    counts = {}
    for i in range(0, len(text) - 10, 5):
        key = text[i : i + 10]
        counts[key] = counts.get(key, 0) + 1
    return counts


def slice_of_slice(text: str) -> str:
    inner = text[1:]
    return inner[5:30] + text[:0]


# ---------------------------------------------------------------- vectorize
def domain(email: str) -> str:
    return email[email.find('@') + 1 :].lower()


def weighted_length(text: str, weight: float) -> float:
    return len(text.strip()) * weight
