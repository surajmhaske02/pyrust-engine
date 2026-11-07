"""Functions under test for 0.7.3 (they need real source files to compile)."""
import math


def add_ints(a, b):
    return a + b


def add_floats(a: float, b: float) -> float:
    return a + b


def both_positive(a: float, b: float) -> bool:
    return a > 0 and b > 0


def nothing(a):
    for i in range(a):
        pass


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


def root_sum(n):
    t = 0.0
    for i in range(n):
        t += math.sqrt(i)
    return t


def helper(x):
    return x * 2


def calls_helper(n):
    t = 0
    for i in range(n):
        t += helper(i)
    return t


def popcounts(n):
    t = 0
    for i in range(-n, n):
        t += (i * 7919).bit_count()
    return t


def joined(words: list[str]) -> str:
    return ','.join(words)


def split_on_char(text: str) -> list[str]:
    return '-'.join(text.split()).split('-')


# ------------------------------------------------------------- Series arguments
def total_length(names):
    t = 0
    for i in range(len(names)):
        t += len(names[i])
    return t


def count_with_at(emails):
    n = 0
    for e in emails:
        if e.find('@') >= 0:
            n += 1
    return n


def last_name(names):
    return names[-1]


def first_sorted(names):
    return sorted(names)[0] + '|' + max(names) + '|' + ' '.join(names)


def contains_a(names):
    return 'a' in names


def plus(names):
    return names + names


def series_sum(values):
    t = 0.0
    for i in range(len(values)):
        t += float(values[i])
    return t


def int_sum(values):
    t = 0
    for v in values:
        t += v
    return t


def last_value(values):
    return values[-1]


# ------------------------------------------------------------- vectorize
def domain_length(email: str) -> int:
    at = email.find('@')
    return len(email) - at - 1 if at >= 0 else 0


def flag_score(active: bool, x: float) -> float:
    return x * 2.0 if active else -x


def flip(active: bool) -> bool:
    return not active


def clean(text: str) -> str:
    return text.strip().lower()


def bool_sum(a: bool, b: bool) -> int:
    return a + b


def scale32(x):
    return x * 3
