# AI and agents

PyRust can't speed up LLM API calls or GPU inference. Those are waiting and GPU time. It does speed up the CPU work around them: retrieval, text processing, scoring and evaluation. Agents run that work over and over.

## Ready-made functions: `pyrust_engine.ai`

```python
from pyrust_engine.ai import BM25, chunk_text, cosine_top_k, levenshtein, minhash, rouge_l, token_f1, tokenize
```

| Area | Functions | Use for |
|---|---|---|
| Retrieval | `cosine_top_k(query, matrix, k)`, `top_k(scores, k)`, `BM25(docs).search(q, k)` | embedding search (float32 or float64, all CPU cores), keyword search |
| Compact vectors | `hamming_top_k`, `int8_top_k`, `binarize`, `quantize_int8` | binary (32x smaller) and int8 (4x smaller) embeddings |
| rank_bm25 users | `BM25Okapi(corpus)` | same API and the same scores as `rank_bm25.BM25Okapi`, much faster |
| Batches | `batch(fn, items)`, `tokenize_many`, `chunk_many` | many documents on all CPU cores |
| Text | `tokenize`, `chunk_text(text, size, overlap)`, `ngrams`, `levenshtein`, `similarity`, `jaccard` | RAG indexing, fuzzy matching, cleaning |
| Deduplication | `minhash(tokens)`, `minhash_similarity(a, b)` | near-duplicate documents |
| Evaluation | `token_f1`, `exact_match`, `rouge_l` | QA and summarization metrics |

Every function is plain Python compiled by PyRust. Its result is exactly what its Python source returns, and it runs as Python where Rust isn't installed. Text functions are fastest on ASCII text. Other characters are handled exactly, though some steps then run in Python.

### Hybrid search for RAG

```python
from pyrust_engine.ai import BM25, chunk_text, cosine_top_k

chunks = [c for doc in documents for c in chunk_text(doc, 800, 100)]
keyword = BM25(chunks)
vectors = embed(chunks)                          # your embedding model, float32 (n, d)

def search(query, k=5):
    ids_kw, _ = keyword.search(query, 20)
    ids_vec, _ = cosine_top_k(embed([query])[0], vectors, 20)
    fused = {}
    for ranking in (ids_kw, ids_vec):            # reciprocal rank fusion
        for rank, i in enumerate(ranking.tolist()):
            fused[i] = fused.get(i, 0.0) + 1 / (61 + rank)
    return [chunks[i] for i in sorted(fused, key=lambda i: -fused[i])[:k]]
```

All searches take `mask=`, a bool array over the rows, to filter by tenant, user, date or document type:

```python
allowed = tenant_ids == current_tenant          # NumPy bool array, one value per chunk
ids, scores = cosine_top_k(query_vec, vectors, 5, mask=allowed)
ids, scores = keyword.search(query, 5, mask=allowed)
```

The full runnable version is [`examples/rag_retriever.py`](https://github.com/surajmhaske02/pyrust-engine/blob/main/examples/rag_retriever.py).

!!! note "When NumPy is already the right tool"
    A dense matrix–vector product (`matrix @ query`) runs in optimized BLAS. `cosine_top_k` is competitive with it and avoids temporary arrays, but the [benchmarks](benchmarks.md) show both honestly. PyRust's bigger wins are where NumPy can't help: per-token loops, dicts, strings and branching logic.

### Replacing rank_bm25

`rank_bm25` is pure Python: building the index and every query loop over all documents in Python. `BM25Okapi` has the same constructor, methods and attributes, and returns the same scores bit for bit (CI compares them with the real package), so switching is one import:

```python
from pyrust_engine.ai import BM25Okapi          # was: from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)
top = bm25.get_top_n(tokenized_query, corpus, n=5)
```

LangChain's `BM25Retriever` builds a `rank_bm25` index by default; pass ours instead:

```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever(vectorizer=BM25Okapi([d.page_content.split() for d in docs]), docs=docs, k=4)
```

### Keeping an index up to date

```python
index = BM25(chunks)
new_ids = index.add(new_chunks)                  # ids of existing documents never change
index.remove([12, 40])                           # removed ids are never returned again
index.save("index.npz")                          # NumPy file, no pickle
index = BM25.load("index.npz")                   # server start: no re-tokenizing
```

After `add()` and `remove()`, scores are exactly those of a fresh index over the live documents. Add documents in batches: each call rebuilds the scoring tables once (NumPy, fast), and `BM25Okapi` keeps rank_bm25's idf rule (`epsilon` times the average idf) over the live documents.

### Binary and int8 embeddings

Binary codes keep one bit per dimension; int8 codes keep one byte. Search a large first pass with them, then re-rank the top candidates with full float vectors if needed:

```python
from pyrust_engine.ai import binarize, hamming_top_k, int8_top_k, quantize_int8

codes = binarize(vectors)                          # (n, d/8) uint8
ids, distances = hamming_top_k(binarize(query_vec), codes, 100)

q8, scale = quantize_int8(vectors)                 # (n, d) int8 and one scale
ids, scores = int8_top_k(quantize_int8(query_vec, scale)[0], q8, 10)
```

Both read the matrix in place, 8 bytes at a time, on all CPU cores. With NumPy, int8 vectors must be cast to int32 first (`int8 @ int8` overflows), which copies the whole matrix on every query; Hamming distance needs `np.bitwise_count` over a temporary array. Results are exact: distances and integer dot products, ties broken by the lower row index.

### Many documents at once

```python
from pyrust_engine.ai import batch, chunk_many, tokenize_many

tokens = tokenize_many(documents)                  # same as [tokenize(d) for d in documents]
chunks = chunk_many(documents, 800, 100)
embedded = batch(my_cpu_function, items, workers=8)
```

Compiled functions release the GIL while they run, so their work runs in parallel. On free-threaded Python (3.14t) everything does. The speedup with the GIL depends on how much of each call is native work; the [benchmarks](benchmarks.md) show the measured number.

## Your own code: `@optimize` and `@vectorize`

The usual AI-app hot spots now compile:

```python
from pyrust_engine import optimize

@optimize
def keyword_hits(text: str, keywords: frozenset[str]) -> dict[str, int]:
    counts = {}
    for word in text.lower().split():
        word = word.strip('.,!?')
        if word in keywords:
            counts[word] = counts.get(word, 0) + 1
    return counts

@optimize
def trim_to_budget(tokens: list[str], budget: int) -> list[str]:
    out = []
    used = 0
    for t in tokens:
        cost = len(t) // 4 + 1
        if used + cost > budget:
            break
        out.append(t)
        used += cost
    return out
```

## For coding agents

### Command line, with JSON output

```bash
python -m pyrust_engine suggest app/scoring.py --json     # per function: compiles? worth it? what to change
python -m pyrust_engine apply app/scoring.py              # diff that adds @optimize where it compiles
python -m pyrust_engine apply app/scoring.py --write      # edit the file
```

`suggest` reports for each function whether it compiles, whether it has a loop (if not, native code can't help), and the action to take. When something blocks compilation, it gives the line, the code and a rewrite hint. `apply` only decorates functions that compile and have a loop, adds the import, and never writes a file that doesn't parse.

### MCP server

```bash
claude mcp add pyrust -- python -m pyrust_engine mcp
```

Other MCP clients use `python -m pyrust_engine mcp` (or `pyrust-mcp`) as a stdio server command.

| Tool | What it returns |
|---|---|
| `check_code` | per function: compiles or not, parameter/return types, the blocking line, reason and hint |
| `suggest_file` | the `suggest` report for a file |
| `apply_file` | a diff adding `@optimize`; writes only with `write: true` |
| `rust_source` | the generated Rust for a function |
| `benchmark` | plain Python vs PyRust time for one call, and whether the results match |
| `doctor` | whether Rust is installed and working |

`check_code`, `rust_source` and `benchmark` execute the code they are given, so use the server on trusted code.

### Instructions for assistants

[`AGENTS.md`](https://github.com/surajmhaske02/pyrust-engine/blob/main/AGENTS.md) gives coding assistants the workflow, the good and poor candidates, and a table of rewrites that make code compile. [`llms.txt`](https://github.com/surajmhaske02/pyrust-engine/blob/main/llms.txt) is the short version for LLM context.

## Serving

Compiled calls release the GIL. In a FastAPI app, run them in a worker thread so the event loop keeps serving requests:

```python
distance = await asyncio.to_thread(levenshtein, a, b)
```

[`examples/fastapi_app.py`](https://github.com/surajmhaske02/pyrust-engine/blob/main/examples/fastapi_app.py) has search, similarity and a compiled scoring rule. To deploy without Rust on the server, [`examples/docker/Dockerfile`](https://github.com/surajmhaske02/pyrust-engine/blob/main/examples/docker/Dockerfile) compiles the functions in a build stage and ships only the prebuilt libraries.
