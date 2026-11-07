"""Kernels for 0.7.1: lists, str/bytes, pandas Series, random and vectorize."""
import random
from random import gauss, randint

from pyrust_engine import prange


# ----------------------------------------------------------------- lists
def scale(values: list[float], k: float) -> list[float]:
    out = [0.0] * len(values)
    for i in range(len(values)):
        out[i] = values[i] * k
    return out


def double_in_place(values: list[int]) -> None:
    for i in range(len(values)):
        values[i] *= 2


def list_stats(values: list[float]) -> float:
    total = 0.0
    biggest = values[0]
    for x in values:
        total += x
        if x > biggest:
            biggest = x
    return total / len(values) + biggest


def int_list_sum(values: list[int]) -> int:
    return sum(values) + len(values)


def running_max(values: list[int]) -> list[int]:
    out = [0] * len(values)
    best = values[0]
    for i, x in enumerate(values):
        if x > best:
            best = x
        out[i] = best
    return out


def copy_pairs(a: list[float], b: list[float]) -> None:
    for i in range(len(a)):
        a[i] = b[i]


def parallel_squares(values: list[float]) -> list[float]:
    out = [0.0] * len(values)
    for i in prange(len(values)):
        out[i] = values[i] * values[i]
    return out


# --------------------------------------------------------------- strings
def edit_distance(a: str, b: str) -> int:
    m = len(b)
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, len(a) + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        for j in range(m + 1):
            prev[j] = cur[j]
    return prev[m]


def vowels(text: str) -> int:
    count = 0
    for ch in text:
        if ch.lower() in 'aeiou':
            count += 1
    return count


def parse_digits(text: str) -> int:
    total = 0
    for ch in text:
        if ch.isdigit():
            total = total * 10 + (ord(ch) - ord('0'))
    return total


def char_classes(text: str) -> int:
    score = 0
    for i in range(len(text)):
        ch = text[i]
        if ch.isalpha():
            score += 1
        if ch.isspace():
            score += 10
        if ch.isupper():
            score += 100
        if ch.isalnum() and not ch.islower():
            score += 1000
        if ch == '!' or ch >= '~':
            score += 10000
    return score


def count_char(text: str, letters: str) -> int:
    n = 0
    for ch in text:
        if ch in letters and ch not in 'xyz':
            n += 1
    return n


def last_upper(text: str) -> str:
    return text[-1].upper()


def caesar_sum(text: str, shift: int) -> int:
    total = 0
    for ch in text:
        total += ord(chr((ord(ch) + shift) % 1114112))
    return total


def byte_hash(data: bytes) -> int:
    h = 5381
    for b in data:
        h = (h * 33 + b) % 4294967296
    return h + data[-1]


# ---------------------------------------------------------------- pandas
def series_mean(s) -> float:
    total = 0.0
    for i in range(len(s)):
        total += s[i]
    return total / len(s)


def series_last(s) -> float:
    return s.iloc[-1] + s.iloc[0]


def series_label_minus_one(s) -> float:
    return s[-1] + 0.0


def series_int_total(s) -> int:
    total = 0
    for x in s:
        total += x * 2
    return total + sum(s)


# ---------------------------------------------------------------- random
def monte_carlo_pi(n: int) -> float:
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


def dice(n: int) -> int:
    total = 0
    for _ in range(n):
        total += randint(1, 6) + random.randrange(10) + random.randrange(-5, 5)
    return total


def big_ranges(n: int) -> int:
    total = 0
    for _ in range(n):
        total = (total + random.randrange(10**15) + randint(-(2**62), 2**62)) % 1000000007
    return total


def walk(n: int) -> float:
    x = 0.0
    for _ in range(n):
        x += gauss(0.0, 1.0) + random.uniform(-1, 1) + random.uniform(0.5, 2)
    return x


def odd_gauss(n: int) -> float:
    total = 0.0
    for _ in range(2 * n + 1):
        total += random.gauss(1.5, 0.25)
    return total


def pick(values: list[float], n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += random.choice(values)
    return total


def pick_char(text: str, n: int) -> int:
    total = 0
    for _ in range(n):
        total += ord(random.choice(text))
    return total


def random_then_fail(n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += random.random()
    return total / (n - n)


def bad_randint(n: int) -> int:
    total = 0
    for _ in range(n):
        total += randint(5, 1)
    return total


def random_helper() -> float:
    return random.random() * 2.0


def calls_random_helper(n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += random_helper()
    return total


# ------------------------------------------------------------- vectorize
def score(price: float, qty: int) -> float:
    return price * qty * (0.9 if qty > 10 else 1.0)


def is_big(x: float) -> bool:
    return x > 2.0


def clamp(x: int, low: int, high: int) -> int:
    return min(max(x, low), high)


def safe_ratio(a: float, b: float) -> float:
    return a / b
