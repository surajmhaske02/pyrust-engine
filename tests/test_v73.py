"""0.7.3: faster calls, pandas 3 / polars columns, BM25Okapi, binary and int8 search, batches."""
import importlib.util
import math
import os
import random
import sys
import sysconfig
import tempfile
import threading
import unittest

from pyrust_engine import check_rust, optimize, vectorize
from pyrust_engine.compiler import Compiler, Unsupported, kind_of_value
from tests import kernels73 as k

NUMPY = importlib.util.find_spec('numpy') is not None
PANDAS = NUMPY and importlib.util.find_spec('pandas') is not None
POLARS = NUMPY and importlib.util.find_spec('polars') is not None
RANK_BM25 = importlib.util.find_spec('rank_bm25') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    from pyrust_engine import ai

    NUMPY2 = int(np.__version__.split('.')[0]) >= 2
else:  # pragma: no cover
    NUMPY2 = False
if PANDAS:
    import pandas as pd
if POLARS:
    import polars as pl


def fast(fn, **options):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False, **options)


def outcome(fn, *args):
    try:
        return ('ok', fn(*args))
    except Exception as exc:
        return ('raise', type(exc).__name__)


# ------------------------------------------------------------------ call path
class CallPath(unittest.TestCase):
    def test_fast_entry_generated_for_scalars_only(self):
        self.assertIn('pyrust_fast', Compiler(k.add_ints).generate())
        self.assertIn('pyrust_fast', Compiler(k.both_positive).generate())
        self.assertIn('pyrust_fast', Compiler(k.nothing).generate())
        self.assertNotIn('pyrust_fast', Compiler(k.total_length, kinds={'names': 'slist'}).generate())

    def test_results_and_counts(self):
        add = fast(k.add_ints)
        for a, b in ((1, 2), (-5, 5), (2**62, 2**62), (2**63 - 1, 1), (10**30, 1)):
            self.assertEqual(add(a, b), a + b)
        self.assertEqual(add(a=3, b=4), 7)
        with self.assertRaises(TypeError):
            add(1)
        self.assertEqual(fast(k.add_floats)(0.1, 0.2), 0.1 + 0.2)
        self.assertIs(fast(k.both_positive)(1.0, 2.0), True)
        self.assertIs(fast(k.both_positive)(1.0, -2.0), False)
        self.assertIsNone(fast(k.nothing)(5))
        report = add.report()
        self.assertEqual(report['calls'], 7)
        if RUST:
            self.assertEqual(report['backend'], 'rust')
            self.assertEqual(report['fallback_reasons'].get('integer overflow'), 2)  # 2**62 + 2**62, 2**63 - 1 + 1

    def test_failure_status_replays_in_python(self):
        grow = fast(k.grow)
        self.assertEqual(grow(10), 3**10)
        self.assertEqual(grow(100), 3**100)  # overflows i64: the fast entry fails, Python replays
        if RUST:
            self.assertIn('integer overflow', grow.report()['fallback_reasons'])

    def test_strict_budget(self):
        spin = fast(k.spin, max_iterations=1000)
        self.assertEqual(spin(100), k.spin(100))
        self.assertEqual(spin(5000), k.spin(5000))
        if RUST:
            self.assertIn('native iteration budget exceeded', spin.report()['fallback_reasons'])

    @unittest.skipUnless(RUST, 'Rust required')
    def test_long_call_moves_to_thread(self):
        spin = fast(k.spin)
        n = 60_000_000  # over the inline budget: the call continues without a budget
        self.assertEqual(spin(n), (n // 8) * 28 + sum(range(n % 8)))
        report = spin.report()
        self.assertEqual(report['long_calls'], 1)
        self.assertEqual(report['fallback_calls'], 0)

    def test_guards(self):
        self.assertAlmostEqual(fast(k.root_sum)(100), k.root_sum(100))
        f = fast(k.calls_helper)
        self.assertEqual(f(50), k.calls_helper(50))
        original = k.helper.__code__
        try:
            k.helper.__code__ = (lambda x: x * 3).__code__
            self.assertEqual(f(50), sum(i * 3 for i in range(50)))
        finally:
            k.helper.__code__ = original
        self.assertEqual(f(50), k.calls_helper(50))

    def test_threads(self):
        add = fast(k.add_floats)
        errors = []

        def work(seed):
            rng = random.Random(seed)
            for _ in range(2000):
                a, b = rng.random(), rng.random()
                if add(a, b) != a + b:
                    errors.append((a, b))

        threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(add.report()['calls'], 16000)

    def test_bit_count(self):
        f = fast(k.popcounts)
        for n in (0, 1, 50, 1000):
            self.assertEqual(f(n), k.popcounts(n))

    def test_one_character_separator(self):
        for words in (['a', 'b', 'c'], [], ['x'], ['ü', '🙂']):
            self.assertEqual(fast(k.joined)(words), ','.join(words))
        for text in ('a b  c', '', 'one'):
            self.assertEqual(fast(k.split_on_char)(text), k.split_on_char(text))


@unittest.skipUnless(sysconfig.get_config_var('Py_GIL_DISABLED'), 'free-threaded Python only')
class FreeThreading(unittest.TestCase):
    def test_gil_stays_disabled(self):
        add = fast(k.add_ints)
        totals = []
        threads = [threading.Thread(target=lambda: totals.append(sum(add(i, 1) for i in range(500)))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(totals, [sum(range(1, 501))] * 4)
        if NUMPY:
            ai.tokenize_many(['a b', 'c d'] * 20, workers=4)
        self.assertFalse(sys._is_gil_enabled())


# ------------------------------------------------------------ Series arguments
@unittest.skipUnless(PANDAS, 'pandas required')
class PandasArguments(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(kind_of_value(pd.Series(['a', 'b'])), 'sseries')
        self.assertEqual(kind_of_value(pd.Series(['a', 'b'], dtype=object)), 'sseries')
        self.assertIsNone(kind_of_value(pd.Series(['a', None])))
        self.assertIsNone(kind_of_value(pd.Series(['a', 1], dtype=object)))
        self.assertIsNone(kind_of_value(pd.Series(['a'], index=[3])))
        if NUMPY2:
            self.assertEqual(kind_of_value(pd.Series([1.0], dtype=np.float32)), 'fseries')

    def test_str_series(self):
        names = pd.Series(['ann', 'bob', 'ünï', '', 'zed 🙂'])
        emails = pd.Series(['a@b.com', 'nope', 'x@y'])
        for fn, arg in (
            (k.total_length, names),
            (k.count_with_at, emails),
            (k.first_sorted, names),
            (k.last_name, names),
            (k.contains_a, names),
            (k.plus, names),
        ):
            f = fast(fn)
            expected, actual = outcome(fn, arg), outcome(f, arg)
            if isinstance(expected[1], pd.Series):
                self.assertTrue(expected[1].equals(actual[1]))
            else:
                self.assertEqual(actual, expected, fn.__name__)
        self.assertEqual(outcome(fast(k.last_name), names), ('raise', 'KeyError'))
        if RUST:
            f = fast(k.total_length)
            f(names)
            self.assertEqual(f.report()['backend'], 'rust')

    def test_series_rules(self):
        for fn in (k.contains_a, k.plus):
            with self.assertRaises(Unsupported) as caught:
                Compiler(fn, kinds={'names': 'sseries'}).generate()
            self.assertIn('Series of str', str(caught.exception))

    @unittest.skipUnless(NUMPY2, 'NumPy 2 required')
    def test_float32_series(self):
        values = pd.Series(np.array([0.1, 2.5, -3.25, 1e-3], dtype=np.float32))
        self.assertEqual(fast(k.series_sum)(values), k.series_sum(values))
        self.assertEqual(outcome(fast(k.last_value), values), outcome(k.last_value, values))


@unittest.skipUnless(POLARS, 'polars required')
class PolarsArguments(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(kind_of_value(pl.Series([1.0, 2.0])), 'plseries')
        self.assertEqual(kind_of_value(pl.Series([1.0], dtype=pl.Float32)), 'plseries')
        self.assertEqual(kind_of_value(pl.Series([1, 2])), 'iplseries')
        self.assertEqual(kind_of_value(pl.Series([1], dtype=pl.UInt8)), 'iplseries')
        self.assertEqual(kind_of_value(pl.Series(['a'])), 'plsseries')
        self.assertIsNone(kind_of_value(pl.Series([1.0, None])))
        self.assertIsNone(kind_of_value(pl.Series([1], dtype=pl.UInt64)))

    def test_values(self):
        cases = (
            (k.series_sum, pl.Series([0.5, 1.25, 3.0])),
            (k.series_sum, pl.Series([0.1, 0.2], dtype=pl.Float32)),
            (k.int_sum, pl.Series([1, 2, 3], dtype=pl.Int16)),
            (k.int_sum, pl.Series([2**62, 2**62])),
            (k.last_value, pl.Series([1.0, 9.0])),
            (k.total_length, pl.Series(['ab', 'c', ''])),
            (k.last_name, pl.Series(['ab', 'c'])),
            (k.count_with_at, pl.Series(['a@b', 'c'])),
        )
        for fn, arg in cases:
            self.assertEqual(outcome(fast(fn), arg), outcome(fn, arg), (fn.__name__, arg.dtype))


# ------------------------------------------------------------------ vectorize
@unittest.skipUnless(NUMPY, 'NumPy required')
class Columns(unittest.TestCase):
    def same(self, vectorized, fn, *columns):
        expected = [fn(*values) for values in zip(*columns)]
        result = vectorized(*columns)
        values = result.to_list() if hasattr(result, 'to_list') else result.tolist()
        self.assertEqual(values, expected)
        return result

    def test_lists(self):
        self.same(vectorize(k.domain_length), k.domain_length, ['a@b.com', 'x', '@'])
        scorer = vectorize(k.flag_score)
        self.same(scorer, k.flag_score, [True, False, True], [1.0, 2.0, 3.0])
        native(self, scorer)
        self.same(vectorize(k.clean), k.clean, [' Hi ', 'YO', ''])
        self.same(vectorize(k.bool_sum), k.bool_sum, [True, True, False], [True, False, False])

    def test_numpy_bool_and_float32(self):
        active = np.array([True, False, True])
        self.same(vectorize(k.flip), k.flip, active.tolist())
        result = vectorize(k.flip)(active)
        self.assertEqual(result.dtype, np.bool_)
        if NUMPY2:
            x = np.array([0.1, 1.5, -2.0], dtype=np.float32)
            scaled = vectorize(k.scale32)
            result = scaled(x)
            native(self, scaled)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.tolist(), [float(k.scale32(v)) for v in x])

    @unittest.skipUnless(PANDAS, 'pandas required')
    def test_pandas(self):
        df = pd.DataFrame(
            {
                'email': ['a@b.com', 'xyz', 'me@site.org', ''],
                'active': [True, False, True, False],
                'x': [1.5, 2.0, 3.0, -1.0],
            },
            index=[10, 20, 30, 40],
        )
        domain = vectorize(k.domain_length)
        result = self.same(domain, k.domain_length, df['email'])
        self.assertEqual(list(result.index), [10, 20, 30, 40])
        native(self, domain)
        self.same(vectorize(k.flag_score), k.flag_score, df['active'], df['x'])
        cleaner = vectorize(k.clean)
        self.same(cleaner, k.clean, df['email'])
        native(self, cleaner)
        self.same(vectorize(k.domain_length), k.domain_length, df['email'].astype(object))
        missing = pd.Series(['a@b', None])
        self.assertEqual(vectorize(k.clean)(pd.Series([' A', 'b '])).tolist(), ['a', 'b'])
        with self.assertRaises(AttributeError):
            vectorize(k.clean)(missing)  # None.strip(): Python's own error

    @unittest.skipUnless(POLARS, 'polars required')
    def test_polars(self):
        frame = pl.DataFrame({'email': ['a@b.com', 'q', 'zz@y'], 'x': [1.0, 2.0, 3.0], 'on': [True, False, True]})
        domain = vectorize(k.domain_length)
        result = self.same(domain, k.domain_length, frame['email'])
        native(self, domain)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, 'email')
        self.same(vectorize(k.flag_score), k.flag_score, frame['on'], frame['x'])
        self.same(vectorize(k.clean), k.clean, frame['email'])
        mapped = frame.with_columns(pl.col('email').map_batches(vectorize(k.domain_length)).alias('d'))
        self.assertEqual(mapped['d'].to_list(), [5, 0, 1])
        self.same(vectorize(k.domain_length), k.domain_length, pl.Series('e', ['a@b', 'cc']))


# ---------------------------------------------------------------- AI library
def okapi_reference(corpus, query, k1=1.5, b=0.75, epsilon=0.25):
    """rank_bm25's BM25Okapi.get_scores, written out (used when rank_bm25 is not installed)."""
    doc_freqs, nd, doc_len = [], {}, []
    for document in corpus:
        doc_len.append(len(document))
        frequencies = {}
        for word in document:
            frequencies[word] = frequencies.get(word, 0) + 1
        doc_freqs.append(frequencies)
        for word in frequencies:
            nd[word] = nd.get(word, 0) + 1
    avgdl = sum(doc_len) / len(corpus)
    idf, idf_sum, negative = {}, 0.0, []
    for word, freq in nd.items():
        value = math.log(len(corpus) - freq + 0.5) - math.log(freq + 0.5)
        idf[word] = value
        idf_sum += value
        if value < 0:
            negative.append(word)
    eps = epsilon * idf_sum / len(idf)
    for word in negative:
        idf[word] = eps
    score = np.zeros(len(corpus))
    lengths = np.array(doc_len)
    for q in query:
        q_freq = np.array([(d.get(q) or 0) for d in doc_freqs])
        score += (idf.get(q) or 0) * (q_freq * (k1 + 1) / (q_freq + k1 * (1 - b + b * lengths / avgdl)))
    return score


def random_corpus(seed, n=80):
    rng = random.Random(seed)
    words = 'alpha beta gamma delta eps zeta eta theta iota kappa lam mu nu xi'.split()
    corpus = [[rng.choice(words) for _ in range(rng.randint(1, 15))] for _ in range(n)]
    corpus[3] = ['alpha'] * 9  # repeated words
    return corpus


@unittest.skipUnless(NUMPY, 'NumPy required')
class Okapi(unittest.TestCase):
    QUERIES = (['alpha', 'beta'], ['nothing'], ['mu', 'mu', 'alpha'], [], ['xi', 'theta', 'eta', 'iota'])

    def test_same_scores_as_reference(self):
        corpus = random_corpus(1)
        for params in ({}, {'k1': 1.2, 'b': 0.5}, {'k1': 2, 'b': 0.9, 'epsilon': 0.5}):
            index = ai.BM25Okapi(corpus, **params)
            for q in self.QUERIES:
                self.assertTrue(np.array_equal(index.get_scores(q), okapi_reference(corpus, q, **params)), (params, q))
        native(self, ai._okapi_compiled)

    @unittest.skipUnless(RANK_BM25, 'rank_bm25 required')
    def test_drop_in_for_rank_bm25(self):
        from rank_bm25 import BM25Okapi as Reference

        corpus = random_corpus(2)
        corpus[7] = []
        texts = [' '.join(d) for d in corpus]
        for params in ({}, {'k1': 1.2, 'b': 0.3}, {'b': 1.0}):
            ref, ours = Reference(corpus, **params), ai.BM25Okapi(corpus, **params)
            self.assertEqual(ours.idf, ref.idf)
            self.assertEqual((ours.avgdl, ours.average_idf, ours.corpus_size), (ref.avgdl, ref.average_idf, ref.corpus_size))
            self.assertEqual(ours.doc_len, ref.doc_len)
            self.assertEqual(ours.doc_freqs, ref.doc_freqs)
            for q in self.QUERIES:
                self.assertTrue(np.array_equal(ours.get_scores(q), ref.get_scores(q), equal_nan=True), (params, q))
                self.assertEqual(ours.get_top_n(q, texts, 7), ref.get_top_n(q, texts, 7))
                self.assertEqual(ours.get_batch_scores(q, [0, 5, 9]), ref.get_batch_scores(q, [0, 5, 9]))
        split = Reference([t.split() for t in texts])  # rank_bm25's tokenizer= starts a process pool
        self.assertTrue(np.array_equal(ai.BM25Okapi(texts, tokenizer=str.split).get_scores(['beta']), split.get_scores(['beta'])))
        with self.assertRaises(ZeroDivisionError):
            ai.BM25Okapi([])

    def test_add_remove_equals_fresh_index(self):
        corpus = random_corpus(3)
        index = ai.BM25Okapi(corpus)
        index.remove([0, 4, 50])
        extra = [['alpha', 'new'], ['beta', 'beta', 'omega']]
        ids = index.add(extra)
        self.assertEqual(ids.tolist(), [80, 81])
        live = [d for i, d in enumerate(corpus) if i not in (0, 4, 50)] + extra
        id_map = [i for i in range(80) if i not in (0, 4, 50)] + [80, 81]
        fresh = ai.BM25Okapi(live)
        self.assertEqual(index.idf, fresh.idf)
        for q in self.QUERIES + (['new', 'omega'],):
            scores = index.get_scores(q)
            self.assertTrue(np.array_equal(scores[id_map], fresh.get_scores(q)), q)
            self.assertTrue(np.array_equal(scores[[0, 4, 50]], np.zeros(3)))
            self.assertTrue(np.array_equal(scores[id_map], okapi_reference(live, q)))
        found, _ = index.search(['alpha'], 200)
        self.assertFalse({0, 4, 50} & set(found.tolist()))
        self.assertEqual(len(index), 79)
        with self.assertRaises(KeyError):
            index.remove([4])
        with self.assertRaises(IndexError):
            index.remove([500])
        with self.assertRaises(ValueError):
            ai.BM25Okapi([['a'], ['b']]).remove([0, 1])

    def test_save_load(self):
        corpus = random_corpus(4)
        corpus[1] = ['ünï', '🙂', 'tab\tword']
        index = ai.BM25Okapi(corpus, k1=1.3)
        index.remove([2])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'okapi.npz')
            index.save(path)
            again = ai.BM25Okapi.load(path)
            self.assertEqual((again.k1, again.b, again.epsilon), (1.3, 0.75, 0.25))
            for q in self.QUERIES + (['🙂'],):
                self.assertTrue(np.array_equal(again.get_scores(q), index.get_scores(q)))
            self.assertEqual(again.idf, index.idf)
            with self.assertRaises(ValueError):
                ai.BM25.load(path)


@unittest.skipUnless(NUMPY, 'NumPy required')
class Keyword(unittest.TestCase):
    TEXTS = [' '.join(d) for d in random_corpus(5)]

    def test_add_remove_mask(self):
        index = ai.BM25(self.TEXTS)
        index.remove([1, 2])
        new = index.add(['alpha omega', 'beta gamma'])
        live = [t for i, t in enumerate(self.TEXTS) if i not in (1, 2)] + ['alpha omega', 'beta gamma']
        id_map = [i for i in range(len(self.TEXTS)) if i not in (1, 2)] + new.tolist()
        fresh = ai.BM25(live)
        for q in ('alpha', 'beta omega', 'zzz', 'mu mu nu'):
            self.assertTrue(np.array_equal(index.scores(q)[id_map], fresh.scores(q)), q)
        ids, _ = index.search('alpha beta', 100)
        self.assertFalse({1, 2} & set(ids.tolist()))
        mask = np.zeros(index.size, dtype=bool)
        mask[[3, 5, 7, 1]] = True
        ids, scores = index.search('alpha beta gamma', 10, mask=mask)
        self.assertTrue(set(ids.tolist()) <= {3, 5, 7})
        with self.assertRaises(ValueError):
            index.search('alpha', 3, mask=[True])

    def test_unchanged_index_matches_0_7_2_formula(self):
        index = ai.BM25(self.TEXTS)
        docs = [ai.tokenize(t) for t in self.TEXTS]
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        avgdl = float(lengths.mean())
        for q in ('alpha beta', 'theta'):
            expected = np.zeros(len(docs))
            for term in ai.tokenize(q):
                df = sum(1 for d in docs if term in d)
                idf = np.log((len(docs) - df + 0.5) / (df + 0.5) + 1.0)
                for i, d in enumerate(docs):
                    tf = float(d.count(term))
                    if tf:
                        expected[i] += idf * tf * 2.5 / (tf + 1.5 * (1.0 - 0.75 + 0.75 * lengths[i] / avgdl))
            self.assertTrue(np.allclose(index.scores(q), expected, rtol=1e-12, atol=0))

    def test_save_load(self):
        index = ai.BM25(self.TEXTS, k1=1.1, b=0.6)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'bm25.npz')
            index.save(path)
            again = ai.BM25.load(path)
        self.assertEqual((again.k1, again.b), (1.1, 0.6))
        self.assertTrue(np.array_equal(again.scores('alpha beta'), index.scores('alpha beta')))
        self.assertEqual(again.search('mu', 5)[0].tolist(), index.search('mu', 5)[0].tolist())


@unittest.skipUnless(NUMPY, 'NumPy required')
class KernelsCompile(unittest.TestCase):
    """The library's kernels must compile (checked without Rust): a kernel that silently
    ran as Python would still give right answers, only slowly."""

    def test_ai_kernels(self):
        rows = dict(rows='iarray', use_rows='bool', out='iarray')
        Compiler(ai._hamming_rows, kinds=dict(matrix='iarray2', query='iarray', **rows)).generate()
        Compiler(ai._int8_rows, kinds=dict(matrix='iarray2', lanes='iarray', **rows)).generate()
        postings = dict(
            terms='iarray',
            starts='iarray',
            stops='iarray',
            postings_doc='iarray',
            postings_tf='iarray',
            idf='array',
            doc_len='array',
            avgdl='f64',
            k1='f64',
            b='f64',
            out='array',
        )
        Compiler(ai._okapi_kernel, kinds=postings).generate()
        Compiler(ai._bm25_kernel, kinds=postings).generate()

    def test_vectorize_result_kinds(self):
        self.assertEqual(vectorize(k.domain_length).out_kind(('strv',)), 'i64')
        self.assertEqual(vectorize(k.clean).out_kind(('strv',)), 'strv')
        self.assertEqual(vectorize(k.flag_score).out_kind(('bool', 'f64')), 'f64')
        self.assertEqual(vectorize(k.flip).out_kind(('bool',)), 'bool')
        if NUMPY2:
            self.assertEqual(vectorize(k.scale32).out_kind(('f32',)), 'f32')


def native(test, compiled):
    """With Rust, the last call really ran natively."""
    if RUST:
        report = compiled.report()
        test.assertEqual(report['backend'], 'rust', report.get('reason'))
        test.assertGreater(report['native_calls'], 0)


@unittest.skipUnless(NUMPY, 'NumPy required')
class Vectors(unittest.TestCase):
    def test_hamming(self):
        rng = np.random.default_rng(7)
        for n, d in ((0, 8), (1, 1), (40, 8), (60, 13), (33, 48)):
            matrix = rng.integers(0, 256, (n, d), dtype=np.uint8)
            query = rng.integers(0, 256, d, dtype=np.uint8)
            distances = np.unpackbits(matrix ^ query, axis=1).sum(axis=1).astype(np.int64)
            ids, found = ai.hamming_top_k(query, matrix, 10)
            expected = np.lexsort((np.arange(n), distances))[:10]
            self.assertEqual(ids.tolist(), expected.tolist())
            self.assertEqual(found.tolist(), distances[expected].tolist())
            mask = rng.random(n) < 0.4
            ids, found = ai.hamming_top_k(query, matrix, 5, mask=mask)
            allowed = np.flatnonzero(mask)
            expected = allowed[np.lexsort((allowed, distances[allowed]))][:5]
            self.assertEqual(ids.tolist(), expected.tolist())
        # ties and identical rows
        matrix = np.zeros((5, 8), dtype=np.uint8)
        self.assertEqual(ai.hamming_top_k(np.zeros(8, dtype=np.uint8), matrix, 3)[0].tolist(), [0, 1, 2])
        native(self, ai._hamming_compiled)
        sliced = rng.integers(0, 256, (20, 17), dtype=np.uint8)[:, 1:]  # not contiguous
        q = rng.integers(0, 256, 16, dtype=np.uint8)
        d = np.unpackbits(sliced ^ q, axis=1).sum(axis=1)
        self.assertEqual(ai.hamming_top_k(q, sliced, 20)[1].tolist(), sorted(d.tolist()))
        with self.assertRaises(ValueError):
            ai.hamming_top_k(q.astype(np.int8), sliced, 3)

    def test_int8(self):
        rng = np.random.default_rng(8)
        for n, d in ((0, 8), (50, 8), (50, 13), (17, 384)):
            matrix = rng.integers(-128, 128, (n, d), dtype=np.int8)
            query = rng.integers(-128, 128, d, dtype=np.int8)
            scores = matrix.astype(np.int64) @ query.astype(np.int64)
            ids, found = ai.int8_top_k(query, matrix, 10)
            expected = np.lexsort((np.arange(n), -scores))[:10]
            self.assertEqual(ids.tolist(), expected.tolist())
            self.assertEqual(found.tolist(), scores[expected].tolist())
            mask = rng.random(n) < 0.5
            ids, _ = ai.int8_top_k(query, matrix, 5, mask=mask)
            allowed = np.flatnonzero(mask)
            self.assertEqual(ids.tolist(), allowed[np.lexsort((allowed, -scores[allowed]))][:5].tolist())
        native(self, ai._int8_compiled)
        extreme = np.full((3, 16), -128, dtype=np.int8)
        self.assertEqual(ai.int8_top_k(np.full(16, -128, dtype=np.int8), extreme, 1)[1].tolist(), [16 * 128 * 128])

    def test_helpers_and_cosine_mask(self):
        rng = np.random.default_rng(9)
        embeddings = rng.standard_normal((30, 20)).astype(np.float32)
        codes = ai.binarize(embeddings)
        self.assertEqual(codes.shape, (30, 3))
        self.assertTrue(np.array_equal(np.unpackbits(codes, axis=1)[:, :20], (embeddings > 0).astype(np.uint8)))
        q8, scale = ai.quantize_int8(embeddings)
        self.assertEqual(q8.dtype, np.int8)
        self.assertLessEqual(np.abs(q8.astype(np.float64) * scale - embeddings).max(), scale / 2 + 1e-6)
        ids, _ = ai.hamming_top_k(ai.binarize(embeddings[4]), codes, 1)
        self.assertEqual(ids.tolist(), [4])
        mask = np.arange(30) >= 10
        ids, scores = ai.cosine_top_k(embeddings[2], embeddings, 5, mask=mask)
        self.assertTrue((ids >= 10).all())
        full_ids, full_scores = ai.cosine_top_k(embeddings[2], embeddings, 30)
        keep = full_ids >= 10
        self.assertEqual(ids.tolist(), full_ids[keep][:5].tolist())
        with self.assertRaises(ValueError):
            ai.top_k(np.ones(3), 2, mask=np.ones(4, dtype=bool))


@unittest.skipUnless(NUMPY, 'NumPy required')
class Batches(unittest.TestCase):
    TEXTS = ['Hello RAG world!', '', 'ünïcode wörds 🙂 here', 'a b c d e f g h'] * 30

    def test_same_results_in_order(self):
        self.assertEqual(ai.tokenize_many(self.TEXTS, workers=4), [ai.tokenize(t) for t in self.TEXTS])
        self.assertEqual(ai.chunk_many(self.TEXTS, 10, 3, workers=3), [ai.chunk_text(t, 10, 3) for t in self.TEXTS])
        self.assertEqual(ai.batch(len, self.TEXTS, workers=1), [len(t) for t in self.TEXTS])
        self.assertEqual(ai.batch(len, []), [])
        self.assertEqual(ai.batch(len, iter(['ab'])), [2])

    def test_errors_propagate(self):
        with self.assertRaises(ZeroDivisionError):
            ai.batch(lambda x: 1 / x, [1, 2, 0, 4] * 10, workers=4)
        with self.assertRaises(ValueError):
            ai.chunk_many(['x'], 5, 5)


if __name__ == '__main__':
    unittest.main()
