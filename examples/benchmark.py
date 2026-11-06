"""Honest micro-benchmarks: equal results checked, first call reported separately,
median of warm runs, and a per-call overhead measurement for a tiny function.

Run with:  python -m examples.benchmark
"""
import json
import statistics
import time
from pyrust_engine import optimize


@optimize
def branch_sum(n: int) -> int:
    """Independent iterations: LLVM can vectorise this heavily (best case)."""
    total = 0
    for i in range(n):
        if i < 12345:
            total += i * i
        else:
            total += i
    return total


@optimize
def lcg_mix(n: int) -> int:
    """Each iteration depends on the previous one (no vectorisation): a more
    typical lower bound for loop-heavy integer code."""
    x = 12345
    acc = 0
    for i in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        if x % 3 == 0:
            acc += x // 7
        else:
            acc -= i
    return acc


@optimize
def tiny(a: int, b: int) -> int:
    """Too little work to amortise the native call: expect a slowdown."""
    return a + b


def measure(fn, args, repeats=5):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def loop_benchmark(fn, n):
    start = time.perf_counter()
    actual = fn(n)
    first_call = time.perf_counter() - start
    assert actual == fn.python(n), 'native and Python results differ'
    python_time = measure(fn.python, (n,))
    wrapped_time = measure(fn, (n,))
    report = fn.report()
    return {'n': n, 'first_call_seconds': round(first_call, 6),
            'python_median_seconds': round(python_time, 6),
            'wrapped_median_seconds': round(wrapped_time, 6),
            'speedup': round(python_time / wrapped_time, 1) if report['backend'] == 'rust' else None,
            'report': report}


def call_overhead(calls=200_000):
    tiny(1, 2)  # compile
    assert tiny(3, 4) == 7
    out = {}
    for label, fn in (('python_ns_per_call', tiny.python), ('wrapped_ns_per_call', tiny)):
        start = time.perf_counter()
        for _ in range(calls):
            fn(3, 4)
        out[label] = round((time.perf_counter() - start) / calls * 1e9)
    out['backend'] = tiny.report()['backend']
    return out


if __name__ == '__main__':
    n = 1_000_000
    print(json.dumps({'branch_sum (vectorisable)': loop_benchmark(branch_sum, n),
                      'lcg_mix (serial dependency)': loop_benchmark(lcg_mix, n),
                      'tiny (per-call overhead)': call_overhead()}, indent=2))
