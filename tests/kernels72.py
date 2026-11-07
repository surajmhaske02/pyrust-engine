"""Kernels for 0.7.2: float32, strings, growable lists, dicts and sets."""
import numpy as np

from pyrust_engine import F32Array, F32Array2, prange

STOPWORDS = frozenset(['the', 'a', 'an', 'of', 'and', 'to', 'in'])
GREETING = 'hello'


# --------------------------------------------------------------- float32
def dot32(a: F32Array, b: F32Array) -> float:
    total = np.float32(0.0)
    for i in range(len(a)):
        total += a[i] * b[i]
    return float(total)


def dot32_in_f64(a: F32Array, b: F32Array) -> float:
    total = 0.0
    for i in range(len(a)):
        total += float(a[i]) * float(b[i])
    return total


def scale32(a: F32Array, k: float) -> F32Array:
    out = np.zeros(len(a), dtype=np.float32)
    for i in range(len(a)):
        out[i] = a[i] * 2.5 + k
    return out


def row_dot(matrix: F32Array2, i: int, query: F32Array):
    s = np.float32(0.0)
    for j in range(matrix.shape[1]):
        s += matrix[i, j] * query[j]
    return s


def row_scores(matrix: F32Array2, query: F32Array, out: F32Array) -> None:
    for i in prange(matrix.shape[0]):
        out[i] = row_dot(matrix, i, query)


def mean32(a: F32Array):
    return sum(a) / np.float32(len(a))


def clip32(a: F32Array, limit: float) -> int:
    count = 0
    for x in a:
        if x > 0.5 or abs(x) >= 2:
            count += 1
    return count


# ---------------------------------------------------------------- strings
def words_upper(text: str) -> list[str]:
    out = []
    for word in text.split():
        if word.isalpha():
            out.append(word.upper())
    return out


def tokenize(text: str) -> list[str]:
    tokens = []
    current = ''
    for ch in text.lower():
        if ch.isalnum():
            current += ch
        elif current:
            tokens.append(current)
            current = ''
    if current:
        tokens.append(current)
    return tokens


def chunk(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(' ', start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def string_tools(text: str, word: str) -> int:
    score = text.find(word) + text.rfind(word) * 10 + text.count(word) * 100
    if text.startswith(word):
        score += 1000
    if text.endswith(word[-2:]):
        score += 2000
    left = text[:5] + text[-3:] + text[2:-2]
    score += len(left) * 3
    score += len(text.replace(word, 'X'))
    score += int(' 42 ') + int(f'{score % 7}')
    return score


def join_words(words: list[str]) -> str:
    kept = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return ' | '.join(kept) + f' ({len(kept)} of {len(words)})'


def parse_numbers(line: str) -> float:
    total = 0.0
    for part in line.split(','):
        part = part.strip()
        if part:
            total += float(part)
    return total


def first_word(text: str) -> str:
    for word in text.split():
        return word
    return ''


def char_list(text: str) -> list[str]:
    return sorted(set(text.replace(' ', '')))


def greet(name: str) -> str:
    return GREETING + ', ' + name + '!' * 3


# ------------------------------------------------------------------ lists
def squares(n: int) -> list[int]:
    return [i * i for i in range(n) if i % 3 != 0]


def evens_then_sort(values: list[float]) -> list[float]:
    out = []
    for x in values:
        if x > 0:
            out.append(x)
    out.sort(reverse=True)
    out.append(-1.0)
    last = out.pop()
    out.append(last * 2)
    return out


def sorted_words(text: str) -> list[str]:
    return sorted(text.split(), reverse=True)


def list_ops(values: list[int]) -> int:
    copy = list(values)
    copy.extend(values[1:3])
    copy.reverse()
    total = copy.index(values[0]) + copy.count(values[0]) * 10 + max(copy) * 100 + min(copy)
    for a, b in zip(values, copy):
        total += a * b
    for i in reversed(range(len(values))):
        total = total * 3 + values[i]
    return total % 1000003


def build_matrix_row(n: int) -> list[float]:
    row = [0.0] * n
    for i in range(n):
        row[i] = i / 2
    return row + [1.5, 2.5]


# ------------------------------------------------------------- dicts/sets
def word_counts(text: str) -> dict[str, int]:
    counts = {}
    for word in text.lower().split():
        if word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    return counts


def top_word(text: str) -> str:
    counts = {}
    for word in text.split():
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return max(counts, key=counts.get)


def bm25_like(query: list[str], doc: list[str], idf: dict[str, float]) -> float:
    tf = {}
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    score = 0.0
    for word in query:
        if word in tf:
            f = tf[word]
            score += idf.get(word, 0.0) * f * 2.2 / (f + 1.2)
    return score


def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def dedupe(words: list[str]) -> list[str]:
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def dict_loops(text: str) -> int:
    counts = {'start': 0}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    total = 0
    for key, value in counts.items():
        total += len(key) * value
    for key in counts:
        total += counts[key]
    for value in counts.values():
        total += value
    counts.pop('start')
    del counts[text[0]]
    total += len(counts) + counts.setdefault('zz', 5)
    return total


def grow_while_looping(n: int) -> int:
    d = {0: 1}
    for k in d:
        d[k + 1] = k
        if len(d) > n:
            break
    return len(d)


def int_keys(values: list[int]) -> dict[int, float]:
    out = {}
    for v in values:
        out[v % 5] = out.get(v % 5, 0.0) + 0.5
    return out


def set_ops(a: list[str], b: list[str]) -> set[str]:
    sa = set(a)
    sb = set(b)
    sa.discard('x')
    return (sa - sb) | (sb ^ {'q'})


def letters(text: str) -> int:
    seen = set()
    for ch in text:
        seen.add(ch)
    return len(seen)


def lookup(table: dict[str, int], keys: list[str]) -> int:
    total = 0
    for k in keys:
        total += table.get(k, -1)
        if k in table:
            total += table[k]
    return total


def helper_counts(words: list[str], counts: dict[str, int]) -> None:
    for w in words:
        counts[w] = counts.get(w, 0) + 1


def uses_helper(text: str) -> dict[str, int]:
    counts = {}
    helper_counts(text.split(), counts)
    return counts
