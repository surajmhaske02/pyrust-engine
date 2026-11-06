---
name: Bug report
about: A result differs from plain Python, a crash, or something that should compile but doesn't
labels: bug
---

**Smallest function that shows the problem**

```python
from pyrust_engine import optimize

@optimize
def f(n: int) -> int:
    ...
```

**Arguments:** `f(...)`

**Plain Python result:** (`f.python(...)`)

**PyRust result:**

**`print(f.explain())` and `f.report()`:**

```text

```

**Environment:** OS, Python version (`python --version`), Rust version (`rustc --version`), PyRust version (`pyrust_engine.__version__`), NumPy version if arrays are involved.
