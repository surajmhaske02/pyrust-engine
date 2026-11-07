"""0.7.1: Python lists, str and bytes, pandas Series, the random module and vectorize."""
import importlib.util
import random
import unittest
import warnings

from pyrust_engine import check_rust, optimize, prange, vectorize
from pyrust_engine.compiler import Compiler, Unsupported, kind_of_value
from tests import kernels71 as k

NUMPY = importlib.util.find_spec('numpy') is not None
PANDAS = importlib.util.find_spec('pandas') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np
if PANDAS:
    import pandas as pd


def plain(value):
    """NumPy scalars (what Python gets from arrays and Series) compared as Python numbers."""
    if NUMPY and isinstance(value, np.generic):
        return value.item()
    return value


def outcome(fn, args):
    try:
        return ('ok', repr(plain(fn(*args))))
    except Exception as exc:
        return ('raise', type(exc).__name__)


def fast(fn):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False)


class Compile(unittest.TestCase):
    """Pure-Python checks: run with or without Rust."""

    def test_new_kernels_compile(self):
        for name, kinds, expected in (
            ('scale', None, 'flist'),
            ('double_in_place', None, 'none'),
            ('list_stats', None, 'f64'),
            ('int_list_sum', None, 'i64'),
            ('running_max', None, 'ilist'),
            ('parallel_squares', None, 'flist'),
            ('edit_distance', None, 'i64'),
            ('vowels', None, 'i64'),
            ('parse_digits', None, 'i64'),
            ('char_classes', None, 'i64'),
            ('count_char', None, 'i64'),
            ('last_upper', None, 'char'),
            ('caesar_sum', None, 'i64'),
            ('byte_hash', None, 'i64'),
            ('series_mean', {'s': 'series'}, 'f64'),
            ('series_last', {'s': 'series'}, 'f64'),
            ('series_int_total', {'s': 'iseries'}, 'i64'),
            ('monte_carlo_pi', None, 'f64'),
            ('dice', None, 'i64'),
            ('big_ranges', None, 'i64'),
            ('walk', None, 'f64'),
            ('odd_gauss', None, 'f64'),
            ('pick', None, 'f64'),
            ('pick_char', None, 'i64'),
            ('calls_random_helper', None, 'f64'),
        ):
            compiler = Compiler(getattr(k, name), kinds=kinds)
            compiler.generate()
            self.assertEqual(compiler.return_type, expected, name)

    def test_random_flag(self):
        for name, uses in (('dice', True), ('calls_random_helper', True), ('vowels', False)):
            compiler = Compiler(getattr(k, name))
            source = compiler.generate()
            self.assertEqual(compiler.program['random'], uses, name)
            self.assertEqual('rng: *mut u32, gauss: *mut f64)' in source, uses, name)

    def test_rules_with_hints(self):
        def writes_series(s):
            for i in range(len(s)):
                s[i] = 1.0

        def int_into_float_list(values: list[float]) -> None:
            for i in range(len(values)):
                values[i] = 1

        def float_list_sum(values: list[float]) -> float:
            return sum(values) + 0.0

        def random_in_prange(out: list[float]) -> None:
            for i in prange(len(out)):
                out[i] = random.random()

        def string_arithmetic(text: str) -> int:
            n = 0
            for ch in text:
                n += ch + 'a'
            return n

        def number_in(values: list[int]) -> int:
            n = 0
            for x in values:
                if x in 'abc':
                    n += 1
            return n

        def long_constant(text: str) -> int:
            n = 0
            for ch in text:
                if ch == 'ab':
                    n += 1
            return n

        for fn, kinds, text in (
            (writes_series, {'s': 'series'}, 'can only read'),
            (int_into_float_list, None, 'float(x)'),
            (float_list_sum, None, 'total += x'),
            (random_in_prange, None, 'not compiled inside prange'),
            (string_arithmetic, None, 'arithmetic'),
            (number_in, None, 'one character'),
            (long_constant, None, 'one-character'),
        ):
            with self.assertRaises(Unsupported, msg=fn.__name__) as ctx:
                Compiler(fn, kinds=kinds).generate()
            self.assertIn(text, str(ctx.exception), fn.__name__)

    def test_kinds_of_values(self):
        self.assertEqual(kind_of_value([1.0, 2.0]), 'flist')
        self.assertEqual(kind_of_value([]), 'flist')
        self.assertEqual(kind_of_value([1, 2]), 'ilist')
        self.assertIsNone(kind_of_value([1, 2.0]))
        self.assertIsNone(kind_of_value([True, False]))
        self.assertIsNone(kind_of_value([2**70]))
        self.assertEqual(kind_of_value('héllo'), 'str')
        self.assertEqual(kind_of_value(b'abc'), 'bytes')
        self.assertIsNone(kind_of_value(bytearray(b'abc')))

    @unittest.skipUnless(PANDAS, 'pandas required')
    def test_series_kinds(self):
        self.assertEqual(kind_of_value(pd.Series([1.0, 2.0])), 'series')
        self.assertEqual(kind_of_value(pd.Series([1, 2])), 'iseries')
        self.assertEqual(kind_of_value(pd.DataFrame({'a': [1.0, 2.0]})['a']), 'series')
        self.assertIsNone(kind_of_value(pd.Series([1.0], index=[5])))
        self.assertIsNone(kind_of_value(pd.Series([1, None], dtype='Int64')))
        self.assertIsNone(kind_of_value(pd.Series(['a', 'b'])))

    def test_fallback_gives_python_results(self):
        # Without Rust (or for values that cannot compile) results are Python's own.
        values = [3, 1, 2]
        k.double_in_place(values)
        f = fast(k.double_in_place)
        other = [3, 1, 2]
        f(other)
        self.assertEqual(values, other)
        self.assertEqual(fast(k.vowels)('Hello'), 2)

    @unittest.skipUnless(NUMPY, 'NumPy required')
    def test_vectorize_python_paths(self):
        v = vectorize(k.score)
        self.assertEqual(v(2.0, 3), k.score(2.0, 3))  # only scalars: a plain call
        with self.assertRaises(ValueError):
            v(np.array([1.0, 2.0]), np.array([1, 2, 3]))
        mixed = v([1, 2.5], 4)  # a mixed list runs the Python loop
        self.assertEqual(mixed.tolist(), [k.score(1, 4), k.score(2.5, 4)])
        small = v(np.array([1, 2], dtype=np.int32), 20)
        self.assertEqual(small.tolist(), [k.score(np.int32(1), 20), k.score(np.int32(2), 20)])
        self.assertIs(optimize.vectorize, vectorize)


@unittest.skipUnless(RUST, 'Rust toolchain required')
class Native(unittest.TestCase):
    def assert_same(self, fn, *cases, kinds_ok=True):
        f = fast(fn)
        for args in cases:
            self.assertEqual(outcome(f, args), outcome(fn, args), (fn.__name__, args))
        if kinds_ok:
            self.assertTrue(f.report()['specializations'], (fn.__name__, f.report()['reason']))
        return f

    def test_lists(self):
        self.assert_same(k.scale, ([1.0, 2.5, -3.0], 2.0), ([], 3.0), ([1e308], 10.0))
        self.assert_same(k.list_stats, ([1.0, 5.0, 2.0],), ([-1.5],))
        self.assert_same(k.int_list_sum, ([1, 2, 3],), ([],), ([2**62, 2**62],))
        self.assert_same(k.running_max, ([3, 1, 4, 1, 5],), ([7],))
        self.assert_same(k.parallel_squares, ([float(i) for i in range(10_000)],))
        with self.assertRaises(IndexError):
            fast(k.list_stats)([])

    def test_list_written_in_place(self):
        f = fast(k.double_in_place)
        values = [1, -2, 2**61]
        same = values
        self.assertIsNone(f(values))
        self.assertIs(values, same)
        self.assertEqual(values, [2, -4, 2**62])
        self.assertTrue(all(type(x) is int for x in values))
        self.assertEqual(f.report()['backend'], 'rust')
        overflow = [2**62]
        f(overflow)  # 2**63 does not fit: replayed in Python, which uses big ints
        self.assertEqual(overflow, [2**63])

    def test_same_list_twice(self):
        f = fast(k.copy_pairs)
        a = [1.0, 2.0]
        f(a, a)
        self.assertEqual(a, [1.0, 2.0])
        self.assertIn('same list', f.report()['fallback_reasons'].__repr__())
        b = [0.0, 0.0]
        f(b, [3.0, 4.0])
        self.assertEqual(b, [3.0, 4.0])

    def test_strings(self):
        self.assert_same(k.edit_distance, ('kitten', 'sitting'), ('', 'abc'), ('héllo wörld', 'hello world'), ('🙂🙂', '🙂'))
        self.assert_same(k.vowels, ('Hello World AEIOU',), ('',))
        self.assert_same(k.vowels, ('Ünïcode',), kinds_ok=False)  # non-ASCII lower() replays
        self.assert_same(k.parse_digits, ('a1b2c3',), ('x9y8',))
        self.assert_same(k.parse_digits, ('٣4',), kinds_ok=False)  # Arabic-Indic digit replays
        self.assert_same(k.char_classes, ('Hello World! 42\t\x1c~',))
        self.assert_same(k.count_char, ('banana xyz', 'anx'), ('', 'a'))
        self.assert_same(k.last_upper, ('hello',), ('',))
        self.assert_same(k.caesar_sum, ('abc', 3), ('\U0010ffff', 1), ('a\udc80b', 5))
        self.assert_same(k.byte_hash, (b'hello world',), (bytes(range(256)),), (b'',))

    @unittest.skipUnless(PANDAS, 'pandas required')
    def test_series(self):
        s = pd.Series([1.0, 2.5, 4.0])
        self.assert_same(k.series_mean, (s,), (pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})['b'],))
        self.assert_same(k.series_last, (s,))
        self.assert_same(k.series_label_minus_one, (s,), kinds_ok=False)  # KeyError in both
        self.assert_same(k.series_int_total, (pd.Series([1, 2, 3]),), (pd.Series([], dtype='int64'),))
        f = fast(k.series_mean)
        with self.assertRaises(KeyError):
            f(pd.Series([2.0, 4.0], index=[10, 20]))  # labels 0 and 1 do not exist: Python's KeyError
        self.assertIn('0..n-1 index', ' '.join(f.report()['fallback_reasons']))

    def check_random(self, fn, *cases):
        f = fast(fn)
        for seed in (0, 1, 12345):
            for args in cases:
                random.seed(seed)
                expected = outcome(fn, args)
                state = random.getstate()
                random.seed(seed)
                got = outcome(f, args)
                self.assertEqual(got, expected, (fn.__name__, seed, args))
                self.assertEqual(random.getstate(), state, (fn.__name__, seed, args))
        return f

    def test_random_exact(self):
        for fn, cases in (
            (k.monte_carlo_pi, [(1,), (1000,), (5000,)]),
            (k.dice, [(0,), (700,)]),
            (k.big_ranges, [(300,)]),
            (k.walk, [(1,), (333,)]),
            (k.odd_gauss, [(0,), (10,)]),
            (k.pick, [([1.0, 2.0, 3.5], 500)]),
            (k.pick_char, [('abcdéf🙂', 200)]),
            (k.calls_random_helper, [(400,)]),
        ):
            f = self.check_random(fn, *cases)
            self.assertEqual(f.report()['backend'], 'rust', fn.__name__)

    def test_random_continues_python_sequence(self):
        f = fast(k.monte_carlo_pi)
        random.seed(7)
        random.random()
        random.gauss(0, 1)  # leaves a cached gauss value in the state
        snapshot = random.getstate()
        value = k.monte_carlo_pi(100)
        after = random.random()
        random.setstate(snapshot)
        self.assertEqual(f(100), value)
        self.assertEqual(random.random(), after)

    def test_random_replays(self):
        self.check_random(k.random_then_fail, (50,))
        self.check_random(k.bad_randint, (3,))
        self.check_random(k.pick, ([], 2))

    def test_random_rebound_falls_back(self):
        f = fast(k.monte_carlo_pi)
        random.seed(3)
        f(10)
        original = random.random
        try:
            random.random = lambda: 0.0
            self.assertEqual(f(10), 4.0)
            self.assertIn('random has been rebound', ' '.join(f.report()['fallback_reasons']))
        finally:
            random.random = original

    @unittest.skipUnless(NUMPY, 'NumPy required')
    def test_vectorize(self):
        prices = np.array([1.0, 2.5, 3.0, -4.0])
        qty = np.array([5, 20, 11, 0])
        v = vectorize(k.score)
        expected = np.array([k.score(p, q) for p, q in zip(prices.tolist(), qty.tolist())])
        self.assertEqual(v(prices, qty).tobytes(), expected.tobytes())
        self.assertEqual(v.report()['backend'], 'rust')
        self.assertEqual(v([1.0, 2.0], 3).tolist(), [k.score(1.0, 3), k.score(2.0, 3)])
        big = vectorize(k.is_big, parallel=True)
        values = np.linspace(0, 4, 50_001)
        self.assertEqual(big(values).tolist(), [x > 2.0 for x in values.tolist()])
        self.assertEqual(big(values).dtype, np.bool_)
        c = vectorize(k.clamp)
        self.assertEqual(c(np.array([-5, 3, 99]), 0, 10).tolist(), [0, 3, 10])
        self.assertEqual(c(np.array([-5, 3, 99]), 0, 10).dtype, np.int64)
        r = vectorize(k.safe_ratio)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got = r(np.array([1.0, 2.0]), np.array([2.0, 0.0]))  # 2/0 replays the loop in Python
            want = np.array([np.float64(1.0) / np.float64(2.0), np.float64(2.0) / np.float64(0.0)])
        self.assertEqual(got.tobytes(), want.tobytes())

    @unittest.skipUnless(NUMPY and PANDAS, 'NumPy and pandas required')
    def test_vectorize_pandas(self):
        df = pd.DataFrame({'p': [1.0, 2.0, 3.0], 'q': [1, 20, 30]}, index=[10, 20, 30])
        out = vectorize(k.score)(df['p'], df['q'])
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(list(out.index), [10, 20, 30])
        self.assertEqual(out.tolist(), [k.score(1.0, 1), k.score(2.0, 20), k.score(3.0, 30)])


if __name__ == '__main__':
    unittest.main()
