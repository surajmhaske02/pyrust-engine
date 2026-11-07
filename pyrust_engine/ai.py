"""Ready-made fast functions for AI and LLM applications: retrieval, text processing and evaluation.

    from pyrust_engine.ai import cosine_top_k, BM25, levenshtein, chunk_text, token_f1

Each function is plain Python compiled by PyRust, so it returns exactly what the Python
code returns, runs as plain Python where Rust is not installed, and its source is a
readable reference implementation. Text functions are fastest on ASCII text; other
characters (é, 漢字, emoji) are handled exactly but some steps replay in Python.

Retrieval:   cosine_top_k, top_k, BM25
Text:        tokenize, chunk_text, ngrams, levenshtein, similarity, jaccard
Dedup:       minhash, minhash_similarity
Evaluation:  token_f1, exact_match, rouge_l
"""

from .compiler import F32Array, F32Array2, F64Array, F64Array2, prange
from .runtime import optimize

__all__ = [
    'BM25',
    'chunk_text',
    'cosine_top_k',
    'exact_match',
    'jaccard',
    'levenshtein',
    'minhash',
    'minhash_similarity',
    'ngrams',
    'rouge_l',
    'similarity',
    'token_f1',
    'tokenize',
    'top_k',
]

_FAST = {'warn_on_fallback': False}


def _numpy():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError('this function needs NumPy: pip install "pyrust-engine[arrays]"') from None
    return np


# ------------------------------------------------------------------ text
def _tokenize(text: str) -> list[str]:
    tokens = []
    current = ''
    for ch in text:
        if ch.isalnum():
            current += ch.lower()
        elif current:
            tokens.append(current)
            current = ''
    if current:
        tokens.append(current)
    return tokens


tokenize = optimize(_tokenize, **_FAST)
tokenize.__doc__ = """Lower-case word tokens: runs of letters and digits ("Don't stop!" -> ['don', 't', 'stop'])."""


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(' ', start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


_chunk_compiled = optimize(_chunk_text, **_FAST)


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into pieces of at most `size` characters, breaking at spaces, with
    `overlap` characters repeated between neighbours (for RAG indexing)."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('need size > 0 and 0 <= overlap < size')
    return _chunk_compiled(text, size, overlap)


def _ngrams(tokens: list[str], n: int) -> list[str]:
    out = []
    for i in range(len(tokens) - n + 1):
        gram = tokens[i]
        for j in range(i + 1, i + n):
            gram = gram + ' ' + tokens[j]
        out.append(gram)
    return out


_ngrams_compiled = optimize(_ngrams, **_FAST)


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Consecutive token n-grams joined by spaces: ngrams(['a', 'b', 'c']) -> ['a b', 'b c']."""
    if n < 1:
        raise ValueError('n must be at least 1')
    return _ngrams_compiled(tokens, n)


def _levenshtein(a: str, b: str) -> int:
    m = len(b)
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for j in range(m + 1):
        prev[j] = j
    for i in range(1, len(a) + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        for j in range(m + 1):
            prev[j] = cur[j]
    return prev[m]


levenshtein = optimize(_levenshtein, **_FAST)
levenshtein.__doc__ = """Edit distance: the number of single-character inserts, deletes and substitutions."""


def similarity(a: str, b: str) -> float:
    """1.0 for equal strings, 0.0 for completely different ones (normalized edit distance)."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - levenshtein(a, b) / longest


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return len(sa & sb) / union


jaccard = optimize(_jaccard, **_FAST)
jaccard.__doc__ = """Jaccard similarity of two token lists: shared distinct tokens / all distinct tokens."""


# ----------------------------------------------------------------- dedup
_PRIME = 2147483647  # 2**31 - 1


def _string_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = ((h ^ ord(ch)) * 16777619) % 4294967296
    return h % _PRIME


def _minhash(tokens: list[str], num_perm: int, seed: int) -> list[int]:
    a = [0] * num_perm
    b = [0] * num_perm
    state = seed % _PRIME + 1
    for k in range(num_perm):
        state = state * 48271 % _PRIME
        a[k] = state
        state = state * 48271 % _PRIME
        b[k] = state
    signature = [_PRIME] * num_perm
    for token in tokens:
        h = _string_hash(token)
        for k in range(num_perm):
            value = (a[k] * h + b[k]) % _PRIME
            if value < signature[k]:
                signature[k] = value
    return signature


_minhash_compiled = optimize(_minhash, **_FAST)


def minhash(tokens: list[str], num_perm: int = 128, seed: int = 1) -> list[int]:
    """MinHash signature of a token set; compare two with minhash_similarity (near-duplicates)."""
    return _minhash_compiled(tokens, num_perm, seed)


def minhash_similarity(a: list[int], b: list[int]) -> float:
    """Estimated Jaccard similarity from two MinHash signatures of the same length."""
    if len(a) != len(b) or not a:
        raise ValueError('signatures must be non-empty and the same length')
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


# ------------------------------------------------------------- evaluation
def _token_f1(prediction: str, truth: str) -> float:
    pred = _tokenize(prediction)
    gold = _tokenize(truth)
    if not pred or not gold:
        return 1.0 if len(pred) == len(gold) else 0.0
    counts = {}
    for t in gold:
        counts[t] = counts.get(t, 0) + 1
    same = 0
    for t in pred:
        left = counts.get(t, 0)
        if left > 0:
            same += 1
            counts[t] = left - 1
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


token_f1 = optimize(_token_f1, **_FAST)
token_f1.__doc__ = """SQuAD-style token F1 between a prediction and the reference answer."""


def exact_match(prediction: str, truth: str) -> bool:
    """True when both have the same tokens (case and punctuation ignored)."""
    return tokenize(prediction) == tokenize(truth)


def _rouge_l(candidate: str, reference: str) -> float:
    c = _tokenize(candidate)
    r = _tokenize(reference)
    if not c or not r:
        return 0.0
    m = len(r)
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, len(c) + 1):
        for j in range(1, m + 1):
            if c[i - 1] == r[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        for j in range(m + 1):
            prev[j] = cur[j]
    lcs = prev[m]
    if lcs == 0:
        return 0.0
    precision = lcs / len(c)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


rouge_l = optimize(_rouge_l, **_FAST)
rouge_l.__doc__ = """ROUGE-L F-measure: longest common subsequence of tokens."""


# -------------------------------------------------------------- retrieval
def _row_cosine32(matrix: F32Array2, i: int, query: F32Array, query_norm: float) -> float:
    dot = 0.0
    norm = 0.0
    for j in range(matrix.shape[1]):
        x = float(matrix[i, j])
        dot += x * float(query[j])
        norm += x * x
    if norm == 0.0 or query_norm == 0.0:
        return 0.0
    return dot / ((norm**0.5) * query_norm)


def _cosine_scores32(matrix: F32Array2, query: F32Array, query_norm: float, out: F64Array) -> None:
    for i in prange(matrix.shape[0]):
        out[i] = _row_cosine32(matrix, i, query, query_norm)


def _row_cosine64(matrix: F64Array2, i: int, query: F64Array, query_norm: float) -> float:
    dot = 0.0
    norm = 0.0
    for j in range(matrix.shape[1]):
        x = matrix[i, j]
        dot += x * query[j]
        norm += x * x
    if norm == 0.0 or query_norm == 0.0:
        return 0.0
    return dot / ((norm**0.5) * query_norm)


def _cosine_scores64(matrix: F64Array2, query: F64Array, query_norm: float, out: F64Array) -> None:
    for i in prange(matrix.shape[0]):
        out[i] = _row_cosine64(matrix, i, query, query_norm)


_scores32 = optimize(_cosine_scores32, **_FAST)
_scores64 = optimize(_cosine_scores64, **_FAST)


def top_k(scores, k: int):
    """Indices of the k largest scores, best first (ties: lower index first), and their scores."""
    np = _numpy()
    scores = np.asarray(scores)
    n = scores.shape[0]
    k = max(0, min(int(k), n))
    if k == 0:
        return np.zeros(0, dtype=np.int64), scores[:0]
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        candidates = np.flatnonzero(scores >= scores[part].min())  # keep every tie at the cut-off
    else:
        candidates = np.arange(n)
    order = candidates[np.lexsort((candidates, -scores[candidates]))][:k]
    return order.astype(np.int64), scores[order]


def cosine_top_k(query, matrix, k: int = 10):
    """Top-k rows of `matrix` by cosine similarity to `query` (embedding search).

    query: 1-D array of length d; matrix: 2-D array (n, d), float32 or float64.
    Returns (indices, scores), best first. Rows are scored on all CPU cores; float32
    inputs are accumulated in float64.
    """
    np = _numpy()
    matrix = np.ascontiguousarray(matrix)
    dtype = np.float32 if matrix.dtype == np.float32 else np.float64
    matrix = matrix.astype(dtype, copy=False)
    query = np.ascontiguousarray(query, dtype=dtype)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError('need query of shape (d,) and matrix of shape (n, d)')
    query_norm = float(np.sqrt(np.dot(query.astype(np.float64), query.astype(np.float64))))
    out = np.empty(matrix.shape[0], dtype=np.float64)
    (_scores32 if dtype == np.float32 else _scores64)(matrix, query, query_norm, out)
    return top_k(out, k)


def _term_counts(tokens: list[str]) -> dict[str, int]:
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts


_term_counts_compiled = optimize(_term_counts, **_FAST)


def _bm25_kernel(
    terms,
    starts,
    stops,
    postings_doc,
    postings_tf,
    idf,
    doc_len,
    avgdl: float,
    k1: float,
    b: float,
    out,
) -> None:
    for t in range(len(terms)):
        term = terms[t]
        weight = idf[term]
        for p in range(starts[term], stops[term]):
            doc = postings_doc[p]
            tf = float(postings_tf[p])
            norm = k1 * (1.0 - b + b * doc_len[doc] / avgdl)
            out[doc] += weight * tf * (k1 + 1.0) / (tf + norm)


_bm25_compiled = optimize(_bm25_kernel, **_FAST)


class BM25:
    """Okapi BM25 keyword search over a list of documents.

        index = BM25(documents)                 # list of str
        ids, scores = index.search('fast python', k=5)

    Documents are tokenized with `tokenize` (or pass already-tokenized lists of str).
    Scoring a query touches only the documents that contain its words.
    """

    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        np = _numpy()
        self.k1, self.b = float(k1), float(b)
        docs = [d if isinstance(d, list) else tokenize(d) for d in documents]
        self.size = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.size and self.doc_len.sum() else 1.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, count in _term_counts_compiled(tokens).items():
                postings.setdefault(term, []).append((doc_id, count))
        self.vocabulary = {term: i for i, term in enumerate(postings)}
        lengths = [len(p) for p in postings.values()]
        self.stops = np.cumsum(np.array(lengths, dtype=np.int64)) if lengths else np.zeros(0, dtype=np.int64)
        self.starts = self.stops - np.array(lengths, dtype=np.int64)
        flat = [entry for p in postings.values() for entry in p]
        self.postings_doc = np.array([d for d, _ in flat], dtype=np.int64)
        self.postings_tf = np.array([c for _, c in flat], dtype=np.int64)
        df = np.array(lengths, dtype=np.float64)
        self.idf = np.log((self.size - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query):
        """BM25 score of every document for the query (a str or a list of tokens)."""
        np = _numpy()
        tokens = query if isinstance(query, list) else tokenize(query)
        terms = np.array([self.vocabulary[t] for t in tokens if t in self.vocabulary], dtype=np.int64)
        out = np.zeros(self.size, dtype=np.float64)
        if len(terms):
            _bm25_compiled(
                terms,
                self.starts,
                self.stops,
                self.postings_doc,
                self.postings_tf,
                self.idf,
                self.doc_len,
                self.avgdl,
                self.k1,
                self.b,
                out,
            )
        return out

    def search(self, query, k: int = 10):
        """(document indices, scores) of the k best documents, best first."""
        return top_k(self.scores(query), k)
