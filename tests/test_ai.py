"""pyrust_engine.ai: ready-made functions give the same results as their Python source."""
import importlib.util
import math
import unittest

from pyrust_engine import check_rust

NUMPY = importlib.util.find_spec('numpy') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    from pyrust_engine import ai

TEXTS = [
    'Retrieval-augmented generation (RAG) combines search with a language model.',
    'Fast Python: compile hot loops to Rust and keep exact Python results!',
    'The quick brown fox jumps over the lazy dog. The dog sleeps.',
    '',
    'ünïcode wörds and emoji 🙂 are handled exactly',
    'python python python fast fast code',
]


@unittest.skipUnless(NUMPY, 'NumPy required')
class Library(unittest.TestCase):
    def check(self, compiled, plain, *cases):
        for args in cases:
            self.assertEqual(compiled(*args), plain(*args), (plain.__name__, args))

    def test_text(self):
        self.check(ai.tokenize, ai._tokenize, *[(t,) for t in TEXTS])
        self.assertEqual(ai.tokenize("Don't stop!"), ['don', 't', 'stop'])
        for t in TEXTS:
            self.assertEqual(ai.chunk_text(t * 5, 40, 10), ai._chunk_text(t * 5, 40, 10))
        self.assertEqual(ai.ngrams(['a', 'b', 'c'], 2), ['a b', 'b c'])
        self.assertEqual(ai.ngrams(['a'], 3), [])
        self.check(ai.levenshtein, ai._levenshtein, ('kitten', 'sitting'), ('', 'abc'), ('🙂a', 'a'), (TEXTS[0], TEXTS[1]))
        self.assertEqual(ai.similarity('', ''), 1.0)
        self.check(ai.jaccard, ai._jaccard, (['a', 'b'], ['b', 'c']), ([], []), (ai.tokenize(TEXTS[2]), ai.tokenize(TEXTS[5])))
        with self.assertRaises(ValueError):
            ai.chunk_text('abc', 10, 10)

    def test_dedup(self):
        a, b = ai.tokenize(TEXTS[2]), ai.tokenize(TEXTS[2].replace('lazy', 'sleepy'))
        self.assertEqual(ai.minhash(a, 64, 7), ai._minhash(a, 64, 7))
        estimate = ai.minhash_similarity(ai.minhash(a), ai.minhash(b))
        self.assertTrue(0.5 < estimate < 1.0, estimate)
        self.assertEqual(ai.minhash_similarity(ai.minhash(a), ai.minhash(a)), 1.0)

    def test_evaluation(self):
        pairs = [(TEXTS[0], TEXTS[0]), ('the cat sat', 'a cat sat down'), ('', ''), ('x', ''), (TEXTS[1], TEXTS[5])]
        self.check(ai.token_f1, ai._token_f1, *pairs)
        self.check(ai.rouge_l, ai._rouge_l, *pairs)
        self.assertTrue(ai.exact_match('Hello, World!', 'hello world'))
        self.assertAlmostEqual(ai.token_f1('the cat sat', 'a cat sat down'), 2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))

    def test_top_k(self):
        idx, scores = ai.top_k(np.array([0.1, 0.9, 0.5, 0.9, -1.0]), 3)
        self.assertEqual(idx.tolist(), [1, 3, 2])
        self.assertEqual(scores.tolist(), [0.9, 0.9, 0.5])
        self.assertEqual(ai.top_k(np.array([1.0, 2.0]), 10)[0].tolist(), [1, 0])
        self.assertEqual(len(ai.top_k(np.array([1.0]), 0)[0]), 0)

    def test_cosine_top_k(self):
        rng = np.random.default_rng(1)
        for dtype in (np.float32, np.float64):
            m = rng.standard_normal((500, 24)).astype(dtype)
            m[3] = 0.0  # zero row scores 0
            q = (m[42] + 0.05).astype(dtype)
            idx, scores = ai.cosine_top_k(q, m, 5)
            m64, q64 = m.astype(np.float64), q.astype(np.float64)
            norms = np.linalg.norm(m64, axis=1) * np.linalg.norm(q64)
            ref = np.where(norms == 0, 0.0, m64 @ q64 / np.where(norms == 0, 1.0, norms))
            self.assertEqual(idx[0], 42)
            self.assertTrue(np.allclose(scores, ref[idx], rtol=1e-12, atol=1e-12))
            self.assertEqual(idx.tolist(), ai.top_k(ref, 5)[0].tolist())
        with self.assertRaises(ValueError):
            ai.cosine_top_k(np.zeros(3), np.zeros((2, 4)))

    def test_bm25(self):
        docs = TEXTS + ['fast fast fast', 'python code that is fast']
        index = ai.BM25(docs)
        tokens = [ai.tokenize(d) for d in docs]
        n = len(docs)
        avgdl = sum(map(len, tokens)) / n
        for query in ('fast python', 'dog', 'nothing matches', ''):
            q = ai.tokenize(query)
            expected = []
            for doc in tokens:
                score = 0.0
                for term in q:
                    df = sum(1 for d in tokens if term in d)
                    if df == 0:
                        continue
                    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                    tf = doc.count(term)
                    score += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * len(doc) / avgdl))
                expected.append(score)
            self.assertTrue(np.allclose(index.scores(query), expected, rtol=1e-12, atol=1e-15), query)
            if any(expected):
                ids, _ = index.search(query, 2)
                self.assertEqual(ids.tolist(), ai.top_k(np.array(expected), 2)[0].tolist(), query)

    @unittest.skipUnless(RUST, 'Rust toolchain required')
    def test_native(self):
        ai.levenshtein('abc', 'abd')
        ai.tokenize('abc def')
        ai.token_f1('a b', 'a c')
        for fn in (ai.levenshtein, ai.tokenize, ai.token_f1):
            self.assertEqual(fn.report()['backend'], 'rust', fn.report()['reason'])


if __name__ == '__main__':
    unittest.main()
