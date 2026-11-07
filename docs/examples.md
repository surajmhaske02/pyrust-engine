# Examples

Each example is plain Python with `@optimize` added. Results are identical to running without it.

## Monte Carlo estimate of π (int arithmetic)

```python
from pyrust_engine import optimize

@optimize
def estimate_pi(samples: int, seed: int) -> float:
    state = seed
    inside = 0
    for _ in range(samples):
        state = (state * 1103515245 + 12345) % 2**31     # stays well inside 64 bits
        x = state / 2**31
        state = (state * 1103515245 + 12345) % 2**31
        y = state / 2**31
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / samples
```

If an intermediate value ever didn't fit in 64 bits, the call would replay in Python, so the result is always exactly Python's.

## Moving average into an output array

```python
import numpy as np
from pyrust_engine import optimize

@optimize
def moving_average(values, window: int):
    out = np.zeros(len(values))
    total = 0.0
    for i, x in enumerate(values):
        total += x
        if i >= window:
            total -= values[i - window]
        out[i] = total / min(i + 1, window)
    return out
```

## Mean and variance in one pass (tuples)

```python
@optimize
def mean_var(values):
    n, mean, m2 = 0, 0.0, 0.0
    for x in values:
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return mean, m2 / n
```

## Recursion

```python
@optimize
def count_paths(rows: int, cols: int) -> int:
    if rows == 0 or cols == 0:
        return 1
    return count_paths(rows - 1, cols) + count_paths(rows, cols - 1)
```

## 2-D arrays: matrix multiply

```python
@optimize
def matmul(a, b):
    n, m = a.shape
    p = b.shape[1]
    out = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            total = 0.0
            for k in range(m):
                total += a[i, k] * b[k, j]
            out[i, j] = total
    return out
```

## Parallel: apply a formula to every element

```python
import math
from pyrust_engine import optimize, prange

@optimize
def black_scholes_call(spot, strike, rate: float, vol: float, years: float, out) -> int:
    in_the_money = 0
    for i in prange(len(spot)):
        s, k = spot[i], strike[i]
        d1 = (math.log(s / k) + (rate + 0.5 * vol * vol) * years) / (vol * math.sqrt(years))
        d2 = d1 - vol * math.sqrt(years)
        n1 = 0.5 * (1.0 + math.tanh(d1 * 0.7978845608 * (1.0 + 0.044715 * d1 * d1)))
        n2 = 0.5 * (1.0 + math.tanh(d2 * 0.7978845608 * (1.0 + 0.044715 * d2 * d2)))
        out[i] = s * n1 - k * math.exp(-rate * years) * n2
        if s > k:
            in_the_money += 1
    return in_the_money
```

## Booleans and early exit

```python
@optimize
def is_sorted(values) -> bool:
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            return False
    return True
```

## Text: count words and vowels (strings)

```python
@optimize
def count_words(text: str) -> int:
    words = 0
    inside = False
    for ch in text:
        if ch.isspace():
            inside = False
        elif not inside:
            inside = True
            words += 1
    return words

count_words("PyRust makes   plain Python fast")    # 5
```

## Similarity between two strings (edit distance)

```python
@optimize
def edit_distance(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    cur = [0] * (len(b) + 1)
    for j in range(len(b) + 1):
        prev[j] = j
    for i in range(1, len(a) + 1):
        cur[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        for j in range(len(b) + 1):
            prev[j] = cur[j]
    return prev[len(b)]

edit_distance("kitten", "sitting")    # 3
```

## Monte Carlo with Python's `random`

```python
import random

@optimize
def simulate_portfolio(days: int, start: float) -> float:
    value = start
    for _ in range(days):
        value *= 1.0 + random.gauss(0.0005, 0.01)
    return value

random.seed(1)
simulate_portfolio(252, 100.0)        # identical to plain Python after random.seed(1)
```

## Lists in, list out

```python
@optimize
def moving_average(values: list[float], window: int) -> list[float]:
    out = [0.0] * len(values)
    total = 0.0
    for i in range(len(values)):
        total += values[i]
        if i >= window:
            total -= values[i - window]
        out[i] = total / min(i + 1, window)
    return out
```

## pandas: replace `df.apply` with `vectorize`

```python
from pyrust_engine import vectorize

@vectorize
def shipping_cost(weight: float, distance: float) -> float:
    base = 4.99 if weight < 1.0 else 9.99
    return base + 0.05 * distance * max(weight, 0.5)

# Before: df.apply(lambda r: shipping_cost(r["weight"], r["km"]), axis=1)
df["cost"] = shipping_cost(df["weight"], df["km"])     # one native loop, same values
```
