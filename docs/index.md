# PyRust Engine

**Make your Python code faster by adding one line.**

Put `@optimize` above a function. PyRust turns it into Rust, compiles it and runs it as native machine code. You keep writing normal Python, and **your answers never change**: anything PyRust can't run safely runs as normal Python instead.

```python
import math
from pyrust_engine import optimize

def distance(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)      # helpers are compiled together with their caller

@optimize
def total_distance(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += distance(float(i), float(n - i))
    return total

total_distance(300_000)                  # about 65-110x faster than plain Python in our benchmarks
print(total_distance.explain())          # compiles? if not, which line, why, and how to fix it
```

| What you get | |
|---|---|
| ⚡ Speed | 7x to 120x on calculation-heavy code: loops, math, bit operations, helper functions, recursion, NumPy arrays |
| 🛟 Exact results | big integers, math errors and unsupported code replay in Python with the same result or exception |
| 🧵 Parallel loops | `prange()` uses every CPU core for independent iterations |
| 🧰 Everyday values | lists, strings, dicts, sets, float32 embeddings, pandas and polars columns and `random`, with Python's exact results |
| 🧱 Classes | methods that read and update `self.x` (dataclasses, `__slots__`, plain classes); state is written back after the call |
| 🚀 Cheap calls | CPython calls compiled functions directly: about 0.2 µs per call with numbers, arrays or strings |
| 🤖 AI and agents | ready-made `pyrust_engine.ai` (BM25 and a drop-in `BM25Okapi`, cosine, binary and int8 top-k with filters, chunking, metrics, batches), `suggest`/`apply`, an MCP server |
| 🐼 Whole columns | `@vectorize` runs a one-value function over arrays, pandas or polars columns (numbers, bools, text) in one native loop |
| 🧶 Free-threaded Python | tested on 3.14t and 3.15t; PyRust never turns the GIL back on |
| 🔍 Clear diagnostics | `explain()`, one-time warnings with line numbers and hints, `python -m pyrust_engine doctor` |
| 📦 Easy deployment | `pip install pyrust-engine`; prebuild libraries so servers don't need Rust |

Start with [Getting started](getting-started.md), see what compiles in the [Guide](guide.md), or browse the [Examples](examples.md).
