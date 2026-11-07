from pyrust_engine import F64Array

def integer_work(n:int):
    total=0
    for i in range(n):
        if i%7==0:
            total+=i//3
        else:
            total+=i%11
    return total

def float_work(n:int,x:float):
    value=x
    for i in range(n):
        value=(value*1.000001+0.0001)/1.000002
    return value

def array_energy(a:F64Array):
    total=0.0
    for i in range(len(a)):
        x=float(a[i])
        total+=x*x
    return total

def scalar_add(a:int,b:int):
    return a+b

import math


def math_work(n: int) -> float:
    # sqrt, sin and log in a loop: needs the 0.5.0 math support.
    total = 0.0
    for i in range(1, n):
        x = float(i)
        total += math.sqrt(x) * math.sin(x) + math.log(x)
    return total


def distance(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def calls_work(n: int) -> float:
    # A helper function called inside the loop: needs 0.5.0 calls between functions.
    total = 0.0
    for i in range(n):
        total += distance(float(i), float(n - i))
    return total


def bits_work(n: int) -> int:
    # Integer hashing with shifts, xor, masks and ** (0.5.0 operators).
    mask = 2 ** 32 - 1
    h = 2166136261
    for i in range(n):
        h = ((h ^ (i & 255)) * 16777619) & mask
        h = h ^ (h >> 13) ^ ((h << 5) & mask)
    return h


def fib(n: int) -> int:
    # Recursion (0.6.0): needs the return annotation.
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def smooth(values: F64Array, out: F64Array):
    # Writes into an output array (0.6.0); returns None.
    n = len(values)
    for i in range(n):
        left = float(values[i - 1]) if i > 0 else float(values[i])
        right = float(values[i + 1]) if i < n - 1 else float(values[i])
        out[i] = (left + 2.0 * float(values[i]) + right) / 4.0


import random

import numpy as np


def edit_distance(a: str, b: str) -> int:
    # Strings (0.7.1): Levenshtein distance with two rows made by [0] * n.
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


def list_variance(values: list) -> float:
    # A plain Python list of floats (0.7.1): converted once per call.
    mean = 0.0
    for x in values:
        mean += x
    mean /= len(values)
    spread = 0.0
    for x in values:
        spread += (x - mean) * (x - mean)
    return spread / len(values)


def monte_carlo_pi(n: int) -> float:
    # random.random() (0.7.1): continues Python's own random sequence exactly.
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


def price_score(price: float, qty: int) -> float:
    # A tiny one-value function, run over whole arrays with vectorize (0.7.1).
    return price * qty * (0.9 if qty > 10 else 1.0)


def word_count(text: str) -> dict[str, int]:
    # dict and str.split (0.7.2)
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts


def dot32(a, b):
    # float32 arithmetic with NumPy 2 rules (0.7.2)
    total = np.float32(0.0)
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
