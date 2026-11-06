import json
import statistics
import time
from pyrust_engine import optimize

@optimize
def branch_sum(n: int) -> int:
    total = 0
    for i in range(n):
        if i < 12345:
            total += i * i
        else:
            total += i
    return total

def measure(fn, n, repeats=5):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(n)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)

if __name__ == '__main__':
    n = 1_000_000
    start = time.perf_counter()
    actual = branch_sum(n)
    first_call = time.perf_counter() - start
    assert actual == branch_sum.python(n)
    python_time = measure(branch_sum.python, n)
    wrapped_time = measure(branch_sum, n)
    report = branch_sum.report()
    print(json.dumps({'n': n, 'first_call_seconds': first_call,
                      'python_median_seconds': python_time, 'wrapped_median_seconds': wrapped_time,
                      'speedup': python_time / wrapped_time if report['backend'] == 'rust' else None,
                      'report': report}, indent=2))
