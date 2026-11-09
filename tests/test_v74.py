"""0.7.4: direct CPython entry point, zero-copy string slices, grouped text batches."""
import importlib.util
import os
import pickle
import random
import subprocess
import sys
import threading
import unittest
import warnings
from pathlib import Path

from pyrust_engine import PyRustFallbackWarning, check_rust, optimize, vectorize
from pyrust_engine.compiler import Compiler
from tests import kernels74 as k

ROOT = Path(__file__).resolve().parent.parent
NUMPY = importlib.util.find_spec('numpy') is not None
PANDAS = NUMPY and importlib.util.find_spec('pandas') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    from pyrust_engine import ai
if PANDAS:
    import pandas as pd


def fast(fn, **options):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False, **options)


def outcome(fn, *args, **kwargs):
    try:
        return ('ok', fn(*args, **kwargs))
    except Exception as exc:
        return ('raise', type(exc).__name__)


def direct(test, fn):
    """With Rust, CPython calls the native code directly (no ctypes)."""
    if RUST:
        test.assertEqual(fn.report()['dispatch'], 'direct')


# ------------------------------------------------------------- direct calls
class DirectCalls(unittest.TestCase):
    def test_entry_point_generated(self):
        source = Compiler(k.mix).generate()
        self.assertIn('pyrust_py_call', source)
        self.assertIn('pyrust_py_def', source)
        self.assertNotIn('pyrust_py_call', Compiler(k.pieces, kinds={'text': 'str'}).generate())  # returns a dict

    def test_same_results_as_python(self):
        cases = [
            (k.add_ints, (1, 2)),
            (k.add_ints, (-(2**63), 0)),
            (k.add_ints, (2**63 - 1, 1)),  # overflow: replays in Python
            (k.add_ints, (2**70, 1)),  # big int: not handled natively
            (k.add_ints, (1.5, 2)),  # other types: another specialization
            (k.add_ints, (True, 2)),
            (k.add_ints, ('a', 'b')),
            (k.mix, (1.5, 3, True)),
            (k.mix, (1.5, 3, False)),
            (k.mix, (1.5, 3, 1)),
            (k.is_close, (0.1 + 0.2, 0.3)),
            (k.is_close, (1.0, float('nan'))),
            (k.nothing, (5,)),
            (k.divide, (7, 2)),
            (k.divide, (-7, 2)),
            (k.divide, (1, 0)),  # ZeroDivisionError from Python
            (k.root, (2.0,)),
            (k.root, (-1.0,)),  # ValueError from Python
            (k.fib, (20,)),
        ]
        for fn, args in cases:
            f = fast(fn)
            for _ in range(3):  # first call compiles; later calls take the direct path
                self.assertEqual(outcome(f, *args), outcome(fn, *args), (fn.__name__, args))
        f = fast(k.mix)
        f(1.0, 2, True)
        direct(self, f)

    def test_arguments(self):
        f = fast(k.with_default)
        self.assertEqual(f(1), 11)
        self.assertEqual(f(1, 2), 3)
        self.assertEqual(f(a=1, b=5), 6)
        self.assertEqual(f(1, b=5), 6)
        for bad in ((), (1, 2, 3)):
            with self.assertRaises(TypeError):
                f(*bad)
        with self.assertRaises(TypeError):
            f(1, c=2)
        add = fast(k.add_ints)
        add(1, 2)
        with self.assertRaises(TypeError):
            add(1)
        self.assertEqual(add(*[3, 4]), 7)

    def test_counts(self):
        add = fast(k.add_ints)
        for i in range(1000):
            add(i, i)
        add(2**63 - 1, 1)
        add(1.0, 2.0)
        report = add.report()
        self.assertEqual(report['calls'], 1002)
        if RUST:
            # 1000 + the overflow attempt + the float call ran natively; the overflow replayed
            self.assertEqual((report['native_calls'], report['fallback_calls']), (1002, 1))
            self.assertEqual(report['fallback_reasons'], {'integer overflow': 1})
            self.assertIsNone(report['reason'])
        add.reset()
        add(1, 1)
        self.assertEqual(add.report()['calls'], 1003)

    def test_changed_code_defaults_and_globals(self):
        f = fast(k.with_default)
        self.assertEqual(f(1), 11)
        original = k.with_default.__defaults__
        try:
            k.with_default.__defaults__ = (100,)
            self.assertEqual(f(1), 101)
        finally:
            k.with_default.__defaults__ = original
        g = fast(k.scale)
        self.assertEqual(g(5), 11)
        code = k.helper.__code__
        try:
            k.helper.__code__ = (lambda x: x * 10).__code__
            self.assertEqual(g(5), 51)
        finally:
            k.helper.__code__ = code
        self.assertEqual(g(5), 11)
        saved = k.helper
        try:
            k.helper = lambda x: -x
            self.assertEqual(g(5), -4)
        finally:
            k.helper = saved
        self.assertEqual(g(5), 11)
        r = fast(k.root)
        self.assertEqual(r(4.0), 2.0)
        math_module = k.math
        try:
            k.math = type('fake', (), {'sqrt': staticmethod(lambda x: -1.0)})
            self.assertEqual(r(4.0), -1.0)
        finally:
            k.math = math_module
        self.assertEqual(r(4.0), 2.0)

    def test_budget_and_long_calls(self):
        strict = fast(k.spin, max_iterations=1000)
        self.assertEqual(strict(5000), k.spin(5000))
        self.assertEqual(fast(k.grow)(100), 3**100)
        if RUST:
            self.assertIn('native iteration budget exceeded', strict.report()['fallback_reasons'])
            spin = fast(k.spin)
            spin(10)
            n = 60_000_000
            self.assertEqual(spin(n), (n // 8) * 28 + sum(range(n % 8)))
            self.assertEqual(spin.report()['long_calls'], 1)

    def test_threads(self):
        f = fast(k.mix)
        errors = []

        def work(seed):
            rng = random.Random(seed)
            for _ in range(3000):
                a, b, flag = rng.random(), rng.randint(-50, 50), rng.random() < 0.5
                if f(a, b, flag) != k.mix(a, b, flag):
                    errors.append((a, b, flag))

        threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(f.report()['calls'], 24000)

    def test_pickle_methods_and_metadata(self):
        self.assertIs(pickle.loads(pickle.dumps(k.picklable)), k.picklable)
        self.assertEqual(k.picklable(5, 3), 2)
        self.assertEqual(k.Holder().method(4), 5)
        f = fast(k.with_default)
        self.assertEqual(f.__name__, 'with_default')
        self.assertIs(f.python, k.with_default)
        import inspect

        self.assertEqual(str(inspect.signature(f)), '(a, b=10)')

    def test_warning_points_at_caller(self):
        f = optimize(k.add_ints, native_loop_free=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f(2**70, 1)
        found = [w for w in caught if issubclass(w.category, PyRustFallbackWarning)]
        self.assertTrue(found)
        self.assertEqual(Path(found[0].filename).resolve(), Path(__file__).resolve())

    def test_ctypes_path_still_works(self):
        script = (
            'from tests import kernels74 as k\n'
            'from pyrust_engine import optimize\n'
            'f = optimize(k.mix, native_loop_free=True, warn_on_fallback=False)\n'
            'print(f(1.5, 3, True), f(1.5, 3, False), f(2.0, 2**70, True), f.report()["dispatch"])\n'
        )
        env = dict(os.environ, PYRUST_DIRECT_CALLS='0', PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split()[:3], [repr(k.mix(1.5, 3, True)), repr(k.mix(1.5, 3, False)), repr(k.mix(2.0, 2**70, True))])
        self.assertEqual(out.stdout.split()[3], 'wrapper')


# --------------------------------------------------------------- string views
TEXTS = [
    '',
    'short',
    'exactly 7',
    'The quick brown fox jumps over the lazy dog, again and again and again.',
    'ünïcödé wörds and emoji 🙂🙂🙂 in a longer sentence for slicing tests',
    'x' * 500,
]


class StringViews(unittest.TestCase):
    def test_slices_of_arguments(self):
        for fn in (k.middle, k.grown, k.pieces, k.slice_of_slice):
            f = fast(fn)
            for text in TEXTS:
                self.assertEqual(outcome(f, text), outcome(fn, text), (fn.__name__, text[:20]))

    def test_views_are_used(self):
        source = Compiler(k.middle, kinds={'text': 'str'}).generate()
        self.assertIn('s_view(', source.split('fn compute')[1])


# ------------------------------------------------------------- text columns
@unittest.skipUnless(NUMPY, 'NumPy required')
class TextColumns(unittest.TestCase):
    EMAILS = ['a@b.com', 'x', '', 'Me@Site.ORG', 'ünï@cödé.de', 'long.name.here@' + 'sub.' * 20 + 'example.com']

    def check(self, vectorized, fn, *columns):
        expected = [fn(*values) for values in zip(*(c if isinstance(c, list) else [c] * len(columns[0]) for c in columns))]
        result = vectorized(*columns)
        values = result.tolist()
        self.assertEqual(values, expected)
        text_lists = [c for c in columns if isinstance(c, list)]
        native = all(c and all(s.isascii() for s in c) for c in text_lists)  # an empty list has no type: Python loop
        if RUST and native:
            self.assertEqual(vectorized.report()['backend'], 'rust')

    def test_lists(self):
        ascii_emails = [e for e in self.EMAILS if e.isascii()]
        self.check(vectorize(k.domain), k.domain, ascii_emails)
        self.check(vectorize(k.domain), k.domain, self.EMAILS)  # non-ASCII lower() replays in Python
        self.check(vectorize(k.weighted_length), k.weighted_length, ['  a  ', 'bb', '', ' c'], 1.5)
        self.check(vectorize(k.domain), k.domain, [])

    @unittest.skipUnless(PANDAS, 'pandas required')
    def test_pandas(self):
        column = pd.Series([e for e in self.EMAILS if e.isascii()] * 50)
        result = vectorize(k.domain)(column)
        self.assertEqual(result.tolist(), [k.domain(e) for e in column])
        domain = vectorize(k.domain)
        domain(column)
        if RUST:
            self.assertEqual(domain.report()['backend'], 'rust')

    def test_kernel_reads_views(self):
        v = vectorize(k.domain)
        v(['a@b.com'])
        source = v.rust_source()
        if source:
            self.assertIn('s_view(', source)


# ------------------------------------------------------------- text batches
@unittest.skipUnless(NUMPY, 'NumPy required')
class TextBatches(unittest.TestCase):
    def documents(self, seed, n=300):
        rng = random.Random(seed)
        words = ['alpha', 'Beta', 'GAMMA', 'x1', '2024', 'end.', 'mid-word', "don't", 'ünï', 'naïve', 'tab\tsep', '　wide']
        docs = []
        for _ in range(n):
            ascii_only = rng.random() < 0.8
            pool = [w for w in words if w.isascii()] if ascii_only else words
            docs.append(' '.join(rng.choice(pool) for _ in range(rng.randint(0, 80))))
        docs[0] = ''
        docs[1] = 'no_spaces_at_all_' * 30
        docs[2] = '   leading and trailing   '
        return docs

    def test_tokenize_many(self):
        for seed in (1, 2):
            docs = self.documents(seed)
            expected = [ai.tokenize(d) for d in docs]
            for workers in (1, 4):
                self.assertEqual(ai.tokenize_many(docs, workers=workers), expected)
        self.assertEqual(ai.tokenize_many([]), [])
        self.assertEqual(ai.tokenize_many(['', '']), [[], []])
        if RUST:
            ai.tokenize_many(['only ascii words here'] * 10, workers=1)
            self.assertGreater(ai._tokenize_grouped.report()['native_calls'], 0)

    def test_chunk_many(self):
        docs = self.documents(3, 120) + ['a\xa0b c ' * 50, '\n\n\n', 'word ' * 400]
        for size, overlap in ((50, 10), (200, 0), (7, 6), (1000, 200)):
            expected = [ai.chunk_text(d, size, overlap) for d in docs]
            for workers in (1, 3):
                self.assertEqual(ai.chunk_many(docs, size, overlap, workers=workers), expected, (size, overlap))
        if RUST:
            self.assertGreater(ai._chunk_grouped.report()['native_calls'], 0)
        with self.assertRaises(ValueError):
            ai.chunk_many(['x'], 10, 10)


if __name__ == '__main__':
    unittest.main()
