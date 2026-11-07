# AGENTS.md: PyRust Engine for AI coding assistants

This file is for AI coding assistants (Claude Code, Cursor, Copilot, Codex and others). Part 1 covers using PyRust in someone's project. Part 2 covers changing this repository.

## Part 1: speeding up a user's Python code with PyRust

### What PyRust does

`@optimize` compiles one Python function to Rust and runs it natively. **Results never change.** Anything PyRust can't run natively (unsupported code, big integers, errors, missing Rust) runs as the original Python, with the same result or exception. Adding the decorator is therefore safe; the only question is whether the code gets faster.

### Workflow

1. Find the hot function: `python -m pyrust_engine profile script.py`, or ask the user.
2. Check it:
   - CLI: `python -m pyrust_engine suggest file.py --json`
   - MCP tool: `check_code` (see below)
   - In code: `optimize(fn).explain()`
3. If it compiles and has a loop, add `@optimize` (and `from pyrust_engine import optimize`), or run `python -m pyrust_engine apply file.py --write`.
4. If it doesn't compile, apply the `hint` from the report, or move the blocking line out of the hot function.
5. Verify the results match and measure the speedup:
   - MCP tool: `benchmark`
   - Jupyter: `%pyrust_benchmark fn(args)`
   - In code: `fn.report()['backend'] == 'rust'`

### Good candidates (big speedups)

- Loops doing arithmetic, `math`, bit operations or `random`.
- Loops over NumPy arrays (`float64`, `int64`, `float32`), lists, strings, pandas columns (including pandas 3 `str` columns) and polars Series.
- String processing: tokenizing, edit distance, parsing, chunking.
- Counting and grouping with dicts and sets; building lists with `append`.
- Helper functions called inside a loop, and recursion (annotate `-> int` / `-> float`).
- A tiny function applied to every row: use `@vectorize`, not `@optimize`. It takes NumPy arrays, lists, pandas and polars columns of numbers, bools or str.

### Poor candidates (leave as Python)

- No loop: there is nothing to speed up.
- I/O: network, files, databases, LLM API calls, `async`.
- Code that is already vectorized NumPy (`a @ b`, `np.sum`): it already runs in C.
- Calls into other libraries (pandas methods, sklearn, torch, requests).

### Rewrites that make code compile

| Instead of | Write |
|---|---|
| `print(...)` / logging in the loop | return the value; log outside |
| `try/except` in the hot loop | let errors propagate (they replay in Python) |
| `b = a` (aliasing a list or dict) | `b = list(a)` / `dict(a)`, or keep using `a` |
| looping over a set | `for x in sorted(s)` |
| `d.get(k)` without a default | `d.get(k, 0)` |
| float keys in a dict | int or str keys |
| `x.attr` on objects / methods on classes | pass the numbers in as arguments |
| `df.apply(f, axis=1)`, `s.map(f)`, polars `map_elements(f)` | `@vectorize` on `f`, called with the columns (polars: `map_batches`) |
| `'x' in series`, `series + series` on a str Series | `series.tolist()` first (Series operators work element by element) |
| NumPy slices `a[1:]` in a loop | index loops `for i in range(1, len(a))` |
| float32 mixed with `np.float64` values | `float(x)` to compute in float64 |
| `f"{x:.2f}"` (float formatting) | format outside the compiled function |

Lists, dicts and sets **passed in** are read-only inside compiled code. Build a new one and return it.

### Ready-made functions

`pyrust_engine.ai` provides tested functions for AI apps:

- **Retrieval:** `cosine_top_k`, `top_k`, `BM25` (all take `mask=` filters; indexes have `add`, `remove`, `save`, `load`)
- **rank_bm25 replacement:** `BM25Okapi` (same API, same scores): change `from rank_bm25 import BM25Okapi` to `from pyrust_engine.ai import BM25Okapi`
- **Compact vectors:** `hamming_top_k`, `int8_top_k`, `binarize`, `quantize_int8`
- **Batches:** `batch`, `tokenize_many`, `chunk_many`
- **Text:** `tokenize`, `chunk_text`, `ngrams`, `levenshtein`, `similarity`, `jaccard`
- **Deduplication:** `minhash`, `minhash_similarity`
- **Evaluation:** `token_f1`, `exact_match`, `rouge_l`

### MCP server

```
claude mcp add pyrust -- python -m pyrust_engine mcp
```

| Tool | What it does |
|---|---|
| `check_code` | reports, for each function, whether it compiles; if not, the line, the reason and a hint |
| `suggest_file` | per-function actions for a file |
| `apply_file` | returns a diff; writes the file only with `write: true` |
| `rust_source` | shows the Rust code generated for a function |
| `benchmark` | checks the result matches plain Python and reports the speedup |
| `doctor` | reports whether Rust is installed and working |

`check_code`, `rust_source` and `benchmark` execute the code they are given.

### Deployment

Servers don't need Rust if you ship prebuilt libraries:

```
python -m pyrust_engine build module.py
```

Then ship the generated `__pyrust__/` folder with the code. Compiled calls release the GIL, so `await asyncio.to_thread(fn, ...)` keeps web servers responsive. On free-threaded Python (3.14t) PyRust keeps the GIL off.

## Part 2: working on this repository

### Layout

| Path | Contents |
|---|---|
| `pyrust_engine/compiler.py` | Python AST to Rust: numbers, arrays, control flow, calls, `prange`, `random`, FFI |
| `pyrust_engine/compiler_text.py` | strings, lists, dicts and sets, mixed into the compiler |
| `pyrust_engine/rust_text.py` | Rust helpers that reproduce CPython's string, dict and sort rules |
| `pyrust_engine/kinds.py` | the value kinds shared by compiler and runtime |
| `pyrust_engine/runtime.py` | the decorator, specialization per argument types, ctypes calls, fallback |
| `pyrust_engine/packing.py` | the byte format for lists of str, dicts and sets |
| `pyrust_engine/vectorize.py`, `ai.py`, `advisor.py`, `mcp.py` | `@vectorize`, the AI library, suggest/apply, the MCP server |
| `tests/` | `test_*.py` (unittest) and `kernels*.py` (functions under test) |

### Rules

- **Exactness first.** Generated code must give CPython's result. Otherwise it returns an error status so the call replays in Python. Never approximate.
- **Every feature needs a test** that compares the native result with plain Python (`tests/test_v72.py` shows the pattern), including edge cases: empty inputs, Unicode, NaN, overflow, errors.
- **Lint and types:** `ruff check pyrust_engine`, `ruff format pyrust_engine` (single quotes, line length 120), and `mypy`.
- **Tests:** `python -m unittest discover -s tests`. Native tests are skipped without Rust; CI runs them on Linux, macOS and Windows with Python 3.10–3.15.
- **Unsupported code:** raise `Unsupported` through `self.fail(node, message, hint)` with a hint the user can act on.
