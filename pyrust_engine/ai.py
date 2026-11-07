"""Ready-made fast functions for AI and LLM applications: retrieval, text processing and evaluation.

    from pyrust_engine.ai import cosine_top_k, BM25, levenshtein, chunk_text, token_f1

Each function is plain Python compiled by PyRust, so it returns exactly what the Python
code returns, runs as plain Python where Rust is not installed, and its source is a
readable reference implementation. Text functions are fastest on ASCII text; other
characters (é, 漢字, emoji) are handled exactly but some steps replay in Python.

Retrieval:   cosine_top_k, hamming_top_k, int8_top_k, top_k (all take mask= filters),
             BM25, BM25Okapi (drop-in for rank_bm25), binarize, quantize_int8
Text:        tokenize, chunk_text, ngrams, levenshtein, similarity, jaccard
Batches:     batch, tokenize_many, chunk_many (several threads)
Dedup:       minhash, minhash_similarity
Evaluation:  token_f1, exact_match, rouge_l
"""

import concurrent.futures
import math
import os
import sys

from .compiler import F32Array, F32Array2, F64Array, F64Array2, I64Array, I64Array2, prange
from .runtime import optimize

__all__ = [
    'BM25',
    'BM25Okapi',
    'batch',
    'binarize',
    'chunk_many',
    'chunk_text',
    'cosine_top_k',
    'hamming_top_k',
    'int8_top_k',
    'quantize_int8',
    'tokenize_many',
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


tokenize = optimize(_tokenize, warn_on_fallback=False)
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


_chunk_compiled = optimize(_chunk_text, warn_on_fallback=False)


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


_ngrams_compiled = optimize(_ngrams, warn_on_fallback=False)


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


levenshtein = optimize(_levenshtein, warn_on_fallback=False)
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


jaccard = optimize(_jaccard, warn_on_fallback=False)
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


_minhash_compiled = optimize(_minhash, warn_on_fallback=False)


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
    counts: dict[str, int] = {}
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


token_f1 = optimize(_token_f1, warn_on_fallback=False)
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


rouge_l = optimize(_rouge_l, warn_on_fallback=False)
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


_scores32 = optimize(_cosine_scores32, warn_on_fallback=False)
_scores64 = optimize(_cosine_scores64, warn_on_fallback=False)


def top_k(scores, k: int, mask=None):
    """Indices of the k largest scores, best first (ties: lower index first), and their scores.

    mask: optional bool array; only positions where it is True can be returned.
    """
    np = _numpy()
    scores = np.asarray(scores)
    if mask is not None:
        allowed = _allowed(np, mask, scores.shape[0])
        order, best = top_k(scores[allowed], k)
        return allowed[order], best
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


def _allowed(np, mask, n):
    """Row ids allowed by a bool mask of length n (as int64)."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise ValueError(f'mask must be a bool array of length {n}')
    return np.flatnonzero(mask).astype(np.int64)


def cosine_top_k(query, matrix, k: int = 10, mask=None):
    """Top-k rows of `matrix` by cosine similarity to `query` (embedding search).

    query: 1-D array of length d; matrix: 2-D array (n, d), float32 or float64.
    mask: optional bool array of length n; only rows where it is True are returned
    (filters such as tenant, user or date).
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
    return top_k(out, k, mask)


# ------------------------------------------------- binary and int8 vectors
# Both kernels read the vectors as 64-bit words (8 bytes at a time), so the matrices are
# used in place: no float or int32 copy of the whole matrix per query.
_SIGN = 9223372036854775807  # 2**63 - 1


def _hamming_row(matrix: I64Array2, i: int, query: I64Array) -> int:
    total = 0
    for j in range(matrix.shape[1]):
        x = matrix[i, j] ^ query[j]
        total += (x & _SIGN).bit_count() + (1 if x < 0 else 0)
    return total


def _hamming_rows(matrix: I64Array2, query: I64Array, rows: I64Array, use_rows: bool, out: I64Array) -> None:
    for t in prange(out.shape[0]):
        out[t] = _hamming_row(matrix, rows[t] if use_rows else t, query)


def _int8_row(matrix: I64Array2, i: int, lanes: I64Array) -> int:
    total = 0
    for j in range(matrix.shape[1]):
        word = matrix[i, j]
        for b in range(8):
            v = (word >> (8 * b)) & 255
            total += (v - 256 if v >= 128 else v) * lanes[8 * j + b]
    return total


def _int8_rows(matrix: I64Array2, lanes: I64Array, rows: I64Array, use_rows: bool, out: I64Array) -> None:
    for t in prange(out.shape[0]):
        out[t] = _int8_row(matrix, rows[t] if use_rows else t, lanes)


_hamming_compiled = optimize(_hamming_rows, warn_on_fallback=False)
_int8_compiled = optimize(_int8_rows, warn_on_fallback=False)


def _as_words(np, matrix, query, dtype, what):
    """(matrix as int64 words, query, n): pads rows to a multiple of 8 bytes when needed."""
    matrix = np.asarray(matrix)
    query = np.asarray(query)
    if matrix.dtype != dtype or query.dtype != dtype:
        raise ValueError(f'{what} needs {np.dtype(dtype).name} arrays')
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError('need query of shape (d,) and matrix of shape (n, d)')
    n, d = matrix.shape
    pad = -d % 8
    if pad or not matrix.flags.c_contiguous or sys.byteorder != 'little':
        padded = np.zeros((n, d + pad), dtype=dtype)  # zero bytes change neither result
        padded[:, :d] = matrix
        matrix = padded
        query = np.concatenate([query, np.zeros(pad, dtype=dtype)])
    words = matrix.view(np.int64).reshape(n, (d + pad) // 8)
    if not words.flags.aligned:
        words = words.copy()
    return words, np.ascontiguousarray(query), n


def _rows_for(np, mask, n):
    if mask is None:
        return np.zeros(0, dtype=np.int64), False, n
    rows = _allowed(np, mask, n)
    return rows, True, rows.shape[0]


def hamming_top_k(query, matrix, k: int = 10, mask=None):
    """The k rows of a binary matrix closest to `query` in Hamming distance.

    query: uint8 array of d bytes; matrix: uint8 array (n, d): bits packed 8 per byte, as
    made by binarize() or np.packbits. 32x smaller than float32 embeddings.
    mask: optional bool array of length n; only rows where it is True are returned.
    Returns (indices, distances), nearest first (ties: lower index first).
    """
    np = _numpy()
    words, query, n = _as_words(np, matrix, query, np.uint8, 'hamming_top_k')
    q = query.view(np.int64).copy()  # aligned 64-bit words
    rows, use_rows, count = _rows_for(np, mask, n)
    out = np.empty(count, dtype=np.int64)
    _hamming_compiled(words, q, rows, use_rows, out)
    order, negative = top_k(-out, k)
    return (rows[order] if use_rows else order), -negative


def int8_top_k(query, matrix, k: int = 10, mask=None):
    """Top-k rows of an int8 matrix by exact integer dot product with an int8 `query`.

    For quantized embeddings (4x smaller than float32). The products are summed as
    integers, so there is no overflow and no int32 copy of the matrix.
    mask: optional bool array of length n; only rows where it is True are returned.
    Returns (indices, scores), best first (ties: lower index first).
    """
    np = _numpy()
    words, query, n = _as_words(np, matrix, query, np.int8, 'int8_top_k')
    rows, use_rows, count = _rows_for(np, mask, n)
    out = np.empty(count, dtype=np.int64)
    _int8_compiled(words, np.ascontiguousarray(query.astype(np.int64)), rows, use_rows, out)
    order, best = top_k(out, k)
    return (rows[order] if use_rows else order), best


def binarize(embeddings):
    """Binary codes for float embeddings: bit = value > 0, packed 8 per byte (uint8).

    Works on one vector (d,) or a matrix (n, d); use the result with hamming_top_k.
    """
    np = _numpy()
    return np.packbits(np.asarray(embeddings) > 0, axis=-1)


def quantize_int8(embeddings, scale=None):
    """int8 codes for float embeddings: round(x / scale) clipped to -127..127.

    scale defaults to max(|x|) / 127 over the input. Returns (codes, scale); quantize
    queries with the same scale, then search with int8_top_k.
    """
    np = _numpy()
    x = np.asarray(embeddings, dtype=np.float64)
    if scale is None:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        scale = peak / 127.0 if peak else 1.0
    codes = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return codes, float(scale)


# ------------------------------------------------------------- BM25 indexes
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


_bm25_compiled = optimize(_bm25_kernel, warn_on_fallback=False)


def _okapi_kernel(
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
    # The same operations, in the same order, as rank_bm25's BM25Okapi.get_scores.
    for t in range(len(terms)):
        term = terms[t]
        weight = idf[term]
        for p in range(starts[term], stops[term]):
            doc = postings_doc[p]
            tf = float(postings_tf[p])
            out[doc] += weight * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[doc] / avgdl)))


_okapi_compiled = optimize(_okapi_kernel, warn_on_fallback=False)

_FORMAT = 'pyrust-bm25-1'


def _no_ids(np):
    return np.zeros(0, dtype=np.int64)


class _Postings:
    """Documents stored as term counts, with ids that never change.

    Documents are kept per id (term ids and counts in first-seen order); the per-term
    postings used for scoring are rebuilt with NumPy after add() and remove().
    """

    _keep_one = False

    def _start(self, k1, b):
        np = _numpy()
        self.k1, self.b = float(k1), float(b)
        self.vocabulary: dict[str, int] = {}
        self._terms: list[str] = []
        self._doc_ptr = np.zeros(1, dtype=np.int64)
        self._doc_terms = np.zeros(0, dtype=np.int64)
        self._doc_tfs = np.zeros(0, dtype=np.int64)
        self._doc_first = np.zeros(0, dtype=np.int64)
        self._tokens_seen = 0
        self._lengths = np.zeros(0, dtype=np.int64)
        self._alive = np.zeros(0, dtype=np.bool_)

    # ------------------------------------------------------------ storage
    def _tokens(self, document):
        return document if isinstance(document, list) else tokenize(document)

    def _append(self, token_lists):
        """Store tokenized documents; returns their ids.

        Term ids follow first appearance. Each document stores (term id, count, position
        of the term's first occurrence); counting is done with NumPy on the whole batch.
        """
        np = _numpy()
        vocabulary, terms = self.vocabulary, self._terms
        lengths = np.fromiter(map(len, token_lists), dtype=np.int64, count=len(token_lists))
        flat = [token for tokens in token_lists for token in tokens]
        for term in dict.fromkeys(flat):  # new words, in order of first appearance
            if term not in vocabulary:
                vocabulary[term] = len(terms)
                terms.append(term)
        ids = np.fromiter(map(vocabulary.__getitem__, flat), dtype=np.int64, count=len(flat))
        docs = np.repeat(np.arange(lengths.shape[0], dtype=np.int64), lengths)
        width = max(len(terms), 1)
        keys = docs * width + ids
        order = np.argsort(keys)  # positions of the tokens, grouped by (document, term)
        grouped = keys[order]
        starts = np.flatnonzero(np.concatenate([[True], grouped[1:] != grouped[:-1]])) if grouped.size else _no_ids(np)
        unique = grouped[starts]
        counts = np.diff(np.concatenate([starts, [grouped.shape[0]]]))
        first = np.minimum.reduceat(order, starts) + self._tokens_seen if starts.size else _no_ids(np)
        sizes = np.bincount(unique // width, minlength=lengths.shape[0]).astype(np.int64)
        first_id = self._lengths.shape[0]
        self._doc_terms = np.concatenate([self._doc_terms, (unique % width).astype(np.int64)])
        self._doc_tfs = np.concatenate([self._doc_tfs, counts.astype(np.int64)])
        self._doc_first = np.concatenate([self._doc_first, first.astype(np.int64)])
        self._doc_ptr = np.concatenate([self._doc_ptr, self._doc_ptr[-1] + np.cumsum(sizes)])
        self._lengths = np.concatenate([self._lengths, lengths])
        self._alive = np.concatenate([self._alive, np.ones(lengths.shape[0], dtype=np.bool_)])
        self._tokens_seen += len(flat)
        return np.arange(first_id, first_id + lengths.shape[0], dtype=np.int64)

    def _live_postings(self):
        """Term-major postings of the live documents (doc ids ascending within a term), and
        the live terms in order of first appearance."""
        np = _numpy()
        per_doc = np.diff(self._doc_ptr)
        keep = np.repeat(self._alive, per_doc)
        slots = max(self._lengths.shape[0], 1)
        docs = np.repeat(np.arange(self._lengths.shape[0], dtype=np.int64), per_doc)[keep]
        terms = self._doc_terms[keep]
        order = np.argsort(terms * slots + docs)  # (term, doc) pairs are unique
        df = np.bincount(terms, minlength=len(self._terms)).astype(np.int64)
        stops = np.cumsum(df)
        starts = stops - df
        present = np.flatnonzero(df)
        if present.size:
            first = np.minimum.reduceat(self._doc_first[keep][order], starts[present])
            appearance = present[np.argsort(first)]
        else:
            appearance = present
        return appearance, df, starts, stops, docs[order], self._doc_tfs[keep][order]

    # ------------------------------------------------------------ changes
    def add(self, documents):
        """Add documents (str or token lists); returns their new ids (as an int64 array).

        Ids of existing documents do not change. Add in batches: each call rebuilds the
        scoring tables once.
        """
        ids = self._append([self._tokens(d) for d in documents])
        self._refresh()
        return ids

    def remove(self, ids):
        """Remove documents by id. Their ids are never reused or returned by searches."""
        np = _numpy()
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        n = self._lengths.shape[0]
        if ids.size and (ids.min() < 0 or ids.max() >= n):
            raise IndexError(f'document ids must be between 0 and {n - 1}')
        if ids.size and not self._alive[ids].all():
            raise KeyError(f'already removed: {sorted(set(ids[~self._alive[ids]].tolist()))}')
        if self._keep_one and ids.size and not (self._alive.sum() > np.unique(ids).size):
            raise ValueError('cannot remove every document (BM25Okapi needs at least one)')
        self._alive[ids] = False
        self._refresh()

    def __len__(self):
        """Number of live documents."""
        return int(self._alive.sum())

    def alive(self):
        """Bool array: True for each id that has not been removed."""
        return self._alive.copy()

    # --------------------------------------------------------- save / load
    def save(self, path):
        """Save the index to a .npz file (no pickle). Load it with Class.load(path)."""
        np = _numpy()
        text = ''.join(self._terms)
        np.savez(
            path,
            format=np.array(_FORMAT),
            kind=np.array(type(self).__name__),
            params=np.array(self._params(), dtype=np.float64),
            vocab_text=np.frombuffer(text.encode('utf-32-le', 'surrogatepass'), dtype=np.uint32),
            vocab_lengths=np.array([len(t) for t in self._terms], dtype=np.int64),
            doc_ptr=self._doc_ptr,
            doc_terms=self._doc_terms,
            doc_tfs=self._doc_tfs,
            doc_first=self._doc_first,
            lengths=self._lengths,
            alive=self._alive,
        )

    @classmethod
    def load(cls, path):
        """Load an index written by save(). The file must come from the same class."""
        np = _numpy()
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != _FORMAT or str(data['kind']) != cls.__name__:
                raise ValueError(f'{path} is not a saved {cls.__name__} index')
            index = cls.__new__(cls)
            index._start(1.5, 0.75)
            index._restore(data['params'].tolist())
            text = data['vocab_text'].tobytes().decode('utf-32-le', 'surrogatepass')
            ends = np.cumsum(data['vocab_lengths']).tolist()
            starts = [0, *ends[:-1]]
            index._terms = [text[a:b] for a, b in zip(starts, ends)]
            index.vocabulary = {t: i for i, t in enumerate(index._terms)}
            index._doc_ptr = data['doc_ptr']
            index._doc_terms = data['doc_terms']
            index._doc_tfs = data['doc_tfs']
            index._doc_first = data['doc_first']
            index._tokens_seen = int(data['lengths'].sum())
            index._lengths = data['lengths']
            index._alive = data['alive'].copy()
        index._refresh()
        return index


class BM25(_Postings):
    """Okapi BM25 keyword search over a list of documents.

        index = BM25(documents)                 # list of str
        ids, scores = index.search('fast python', k=5)
        index.add(new_documents); index.remove([3, 7])
        index.save('index.npz'); index = BM25.load('index.npz')

    Documents are tokenized with `tokenize` (or pass already-tokenized lists of str).
    Scoring a query touches only the documents that contain its words. idf is
    log((N - df + 0.5) / (df + 0.5) + 1) (Lucene). Document ids are positions in the
    original list, then the ids returned by add(); removing a document keeps every id.
    """

    def __init__(self, documents=(), k1: float = 1.5, b: float = 0.75):
        self._start(k1, b)
        self._append([self._tokens(d) for d in documents])
        self._refresh()

    def _params(self):
        return [self.k1, self.b]

    def _restore(self, params):
        self.k1, self.b = params

    def _refresh(self):
        np = _numpy()
        _, df, self.starts, self.stops, self.postings_doc, self.postings_tf = self._live_postings()
        live = self._lengths[self._alive]
        self.size = int(self._lengths.shape[0])  # ids run 0..size-1, removed ones included
        self.doc_len = self._lengths.astype(np.float64)
        self.avgdl = float(live.astype(np.float64).mean()) if live.size and live.sum() else 1.0
        n = float(live.size)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query):
        """BM25 score for every id (0.0 for removed documents and documents without the words)."""
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

    def search(self, query, k: int = 10, mask=None):
        """(document ids, scores) of the k best documents, best first.

        mask: optional bool array over ids; only documents where it is True are returned.
        Removed documents are never returned.
        """
        scores = self.scores(query)
        if mask is None and self._alive.all():
            return top_k(scores, k)
        np = _numpy()
        allowed = self._alive if mask is None else self._alive & _check_mask(np, mask, self.size)
        return top_k(scores, k, allowed)


def _check_mask(np, mask, n):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise ValueError(f'mask must be a bool array of length {n}')
    return mask


class BM25Okapi(_Postings):
    """Drop-in replacement for rank_bm25.BM25Okapi, with the same scores, compiled.

        from pyrust_engine.ai import BM25Okapi       # instead of: from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
        bm25.get_scores(tokenized_query)              # NumPy array, same values as rank_bm25
        bm25.get_top_n(tokenized_query, corpus, n=5)

    Same constructor (corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25), the same idf
    (log(N - df + 0.5) - log(df + 0.5), negative values replaced by epsilon * average idf)
    and the same arithmetic, so scores match rank_bm25 exactly. Also has add(), remove(),
    save() and load(); removed documents score 0.0.
    """

    _keep_one = True

    def __init__(self, corpus, tokenizer=None, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self._start(k1, b)
        self.k1, self.b = k1, b  # rank_bm25 keeps the values as given
        self.epsilon = epsilon
        if tokenizer:
            corpus = [tokenizer(document) for document in corpus]
        self._append([list(document) for document in corpus])
        self._refresh()

    def _tokens(self, document):
        return list(document)

    def _params(self):
        return [self.k1, self.b, self.epsilon]

    def _restore(self, params):
        self.k1, self.b, self.epsilon = params

    def _refresh(self):
        np = _numpy()
        appearance, df, self._starts, self._stops, self._pdoc, self._ptf = self._live_postings()
        live = self._lengths[self._alive]
        self.corpus_size = int(live.size)
        self.doc_len = self._lengths.tolist()
        self._doc_len = self._lengths.astype(np.float64)
        self.avgdl = int(live.sum()) / self.corpus_size  # ZeroDivisionError for no documents, like rank_bm25
        # idf in rank_bm25's order (terms by first appearance) so the average is the same float.
        order = appearance.tolist()
        counts = df.tolist()
        idf = [0.0] * len(self._terms)
        negative = []
        idf_sum = 0.0
        n = self.corpus_size
        for term_id in order:
            freq = counts[term_id]
            value = math.log(n - freq + 0.5) - math.log(freq + 0.5)
            idf[term_id] = value
            idf_sum += value
            if value < 0:
                negative.append(term_id)
        self.average_idf = idf_sum / len(order) if order else 0.0
        eps = self.epsilon * self.average_idf
        for term_id in negative:
            idf[term_id] = eps
        self._idf = np.array(idf, dtype=np.float64)
        self._live_terms = order
        self._cached_idf = None
        # Documents without the word add 0 unless k1 * (1 - b + b * len / avgdl) can be 0
        # (then rank_bm25 gets 0/0 = nan); that rare case uses rank_bm25's dense formula.
        with np.errstate(all='ignore'):
            norm = self.k1 * (1 - self.b + self.b * self._doc_len[self._alive] / self.avgdl)
        self._dense = bool(norm.size and not np.all(np.isfinite(norm) & (norm != 0)))

    @property
    def idf(self):
        """{word: idf} for the words of the live documents (like rank_bm25)."""
        if self._cached_idf is None:
            values = self._idf.tolist()
            self._cached_idf = {self._terms[t]: values[t] for t in self._live_terms}
        return self._cached_idf

    @property
    def doc_freqs(self):
        """[{word: count}] per document (built on request; rank_bm25 compatibility)."""
        out = []
        ptr = self._doc_ptr.tolist()
        terms, tfs = self._doc_terms.tolist(), self._doc_tfs.tolist()
        for d in range(len(ptr) - 1):
            out.append({self._terms[terms[p]]: tfs[p] for p in range(ptr[d], ptr[d + 1])})
        return out

    def get_scores(self, query):
        """Scores of every document for a tokenized query (list of str), as a NumPy array."""
        np = _numpy()
        if self._dense:
            return self._dense_scores(np, query)
        vocabulary = self.vocabulary
        terms = np.array([vocabulary[q] for q in query if q in vocabulary], dtype=np.int64)
        out = np.zeros(self._lengths.shape[0], dtype=np.float64)
        if terms.size:
            _okapi_compiled(
                terms,
                self._starts,
                self._stops,
                self._pdoc,
                self._ptf,
                self._idf,
                self._doc_len,
                self.avgdl,
                self.k1,
                self.b,
                out,
            )
        return out

    def _dense_scores(self, np, query):
        score = np.zeros(self._lengths.shape[0])
        doc_len = np.array(self.doc_len)
        idf = self.idf
        freqs = self.doc_freqs
        alive = self._alive.tolist()
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) if live else 0 for doc, live in zip(freqs, alive)])
            with np.errstate(all='ignore'):
                score += (idf.get(q) or 0) * (
                    q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                )
        score[~self._alive] = 0.0
        return score

    def get_batch_scores(self, query, doc_ids):
        """Scores of the given document ids, as a list (like rank_bm25)."""
        assert all(di < self._lengths.shape[0] for di in doc_ids)
        return self.get_scores(query)[list(doc_ids)].tolist()

    def get_top_n(self, query, documents, n: int = 5):
        """The n best documents from `documents` (same order as the corpus), like rank_bm25."""
        assert self._lengths.shape[0] == len(documents), "The documents given don't match the index corpus!"
        np = _numpy()
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def search(self, query, k: int = 10, mask=None):
        """(document ids, scores) of the k best live documents, best first (ties: lower id first)."""
        np = _numpy()
        allowed = self._alive if mask is None else self._alive & _check_mask(np, mask, self._lengths.shape[0])
        return top_k(self.get_scores(query), k, allowed)


# ------------------------------------------------------------ batch processing
def batch(function, items, *, workers=None):
    """[function(x) for x in items], run on several threads; results keep the input order.

    Compiled PyRust functions release the GIL while they run, so their native work runs
    in parallel; on free-threaded Python (3.14t) everything does. Exceptions propagate.
    workers: number of threads (default: CPU count).
    """
    items = list(items)
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(items) < 2:
        return [function(x) for x in items]
    size = max(1, -(-len(items) // (workers * 4)))
    parts = [items[i : i + size] for i in range(0, len(items), size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        done = pool.map(lambda part: [function(x) for x in part], parts)
        return [result for part in done for result in part]


def tokenize_many(texts, *, workers=None):
    """tokenize() for every text, on several threads."""
    return batch(tokenize, texts, workers=workers)


def chunk_many(texts, size: int = 1000, overlap: int = 200, *, workers=None):
    """chunk_text() for every text, on several threads: one list of chunks per text."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('need size > 0 and 0 <= overlap < size')
    return batch(lambda text: _chunk_compiled(text, size, overlap), texts, workers=workers)
