"""0.7.7: methods calling methods, try/except, NumPy slices and small dtypes, direct calls with
lists and str/list results, stats() and logging."""
import copy
import importlib.util
import io
import logging
import math
import unittest
import warnings

import pyrust_engine
from pyrust_engine import check_rust, optimize
from pyrust_engine.compiler import Compiler, Unsupported

NUMPY = importlib.util.find_spec('numpy') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    from tests import kernels77 as k


def fast(fn, **options):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False, **options)


def outcome(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            value = fn(*args)
        except Exception as exc:
            return ('raise', type(exc).__name__)
    if isinstance(value, float) and math.isnan(value):
        return ('ok', 'nan')
    return ('ok', value)


def native(test, fn, dispatch=None):
    if RUST:
        report = fn.report()
        test.assertEqual(report['backend'], 'rust', report['reason'])
        test.assertGreater(report['native_calls'], 0)
        if dispatch:
            test.assertEqual(report['dispatch'], dispatch)


def state(obj):
    return {key: (v.tolist() if hasattr(v, 'tolist') else copy.deepcopy(v)) for key, v in vars(obj).items()}


@unittest.skipUnless(NUMPY, 'NumPy required')
class Methods(unittest.TestCase):
    def same(self, method, make, *args, compiled=True):
        f = fast(method)
        for _ in range(3):
            a, b = make(), make()
            self.assertEqual((outcome(f, a, *args), state(a)), (outcome(method, b, *args), state(b)))
        if compiled:
            native(self, f)
        return f

    def test_methods_calling_methods(self):
        self.same(k.Body.advance, lambda: k.Body(0.3, -1.0), 0.05, 200)
        self.same(k.Body.advance, lambda: k.Fast(0.3, -1.0), 0.05, 200)  # overridden speed()
        self.same(k.Math.total, k.Math, 50)  # recursion through self.fact()
        self.same(k.Math.uses_static, k.Math, 5, compiled=False)  # a staticmethod: Python

    def test_changed_methods(self):
        f = fast(k.Body.advance)
        b = k.Body(0.3, -1.0)
        f(b, 0.1, 10)
        saved = k.Body.speed
        try:
            k.Body.speed = lambda self: 100.0
            a, b2 = k.Body(0.3, -1.0), k.Body(0.3, -1.0)
            self.assertEqual(f(a, 0.1, 10), k.Body.advance(b2, 0.1, 10))
        finally:
            k.Body.speed = saved
        shadow, plain = k.Body(0.3, -1.0), k.Body(0.3, -1.0)
        shadow.speed = lambda: 7.0  # an instance attribute hides the method
        plain.speed = lambda: 7.0
        self.assertEqual(f(shadow, 0.1, 10), k.Body.advance(plain, 0.1, 10))

    def test_list_and_dict_attributes(self):
        self.same(k.Store.scale, lambda: k.Store([1.0, 2.5, -3.0], {'a': 1.5}, ['a']), 2.0)
        self.same(k.Store.score, lambda: k.Store([1.0], {'a': 1.5, 'b': 2.0}, ['a', 'b', 'c', 'a']))


class Exceptions(unittest.TestCase):
    def test_try_raise_assert(self):
        if NUMPY:
            f = fast(k.safe_mean)
            for values, n in (([1.0, 2.0, 3.0], 3), ([1.0, 2.0], 5), ([1.0], 0), ([], 0)):
                self.assertEqual(outcome(f, values, n), outcome(k.safe_mean, values, n))
            f([1.0, 2.0, 3.0], 3)
            native(self, f)
            g = fast(k.checked_sum)
            for n in (10, -1, 10**8, 0):
                self.assertEqual(outcome(g, n), outcome(k.checked_sum, n))
            self.assertEqual(fast(k.with_finally)(100), k.with_finally(100))
            p = fast(k.parse_all)
            for texts in (['1', '2', '3'], ['1', 'x', '3'], [], ['-5', ' 7 ']):
                self.assertEqual(outcome(p, texts), outcome(k.parse_all, texts))

    def test_compile_rules(self):
        def returns_in_finally(n):
            try:
                return n
            finally:
                n += 1

        with self.assertRaises(Unsupported):
            Compiler(returns_in_finally).generate()


@unittest.skipUnless(NUMPY, 'NumPy required')
class NumPy(unittest.TestCase):
    def test_slices_and_rows(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(100)
        m = rng.standard_normal((6, 7))
        for fn, args in ((k.window_sums, (a, 5)), (k.window_sums, (a[:3], 5)), (k.row_norms, (m,)), (k.tail_mean, (a,))):
            f = fast(fn)
            self.assertEqual(outcome(f, *args), outcome(fn, *args))
            self.assertEqual(outcome(f, *args), outcome(fn, *args))
            native(self, f)
        self.assertEqual(outcome(fast(k.tail_mean), a[:0]), outcome(k.tail_mean, a[:0]))

    def test_numpy_functions(self):
        f = fast(k.np_math)
        for x in (2.25, -3.5, 0.0, -0.0, 1e200, float('inf')):
            self.assertEqual(outcome(f, x), outcome(k.np_math, x), x)
        g = fast(k.np_math_loop)
        self.assertEqual(g(1000), k.np_math_loop(1000))
        native(self, g)
        self.assertEqual(outcome(fast(k.sqrt_negative), -1.0), ('ok', 'nan'))

    def test_small_dtypes(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, (20, 30), dtype=np.uint8)
        out_a, out_b = np.zeros_like(img), np.zeros_like(img)
        f = fast(k.threshold)
        self.assertEqual(f(img, out_a, 128), k.threshold(img, out_b, 128))
        self.assertTrue(np.array_equal(out_a, out_b))
        native(self, f)
        values = rng.standard_normal(50)
        mask = rng.random(50) < 0.5
        g = fast(k.masked_total)
        self.assertEqual(g(values, mask), k.masked_total(values, mask))
        native(self, g)
        ints = rng.integers(-(2**31), 2**31 - 1, 100, dtype=np.int32)
        self.assertEqual(fast(k.int32_sum)(ints), k.int32_sum(ints))
        small = np.array([10, 250, 3], dtype=np.uint8)
        a1, a2 = small.copy(), small.copy()
        self.assertEqual(outcome(fast(k.bump), a1, 10), outcome(k.bump, a2, 10))  # OverflowError at 260
        self.assertTrue(np.array_equal(a1, a2))
        self.assertEqual(outcome(fast(k.small_math), np.array([255], dtype=np.uint8)), outcome(k.small_math, np.array([255], dtype=np.uint8)))
        with self.assertRaises(Unsupported) as caught:
            Compiler(k.small_math, kinds={'a': 'u8array'}).generate()
        self.assertIn('wraps around', str(caught.exception))


@unittest.skipUnless(NUMPY, 'NumPy required')
class DirectLists(unittest.TestCase):
    def test_list_arguments(self):
        f = fast(k.list_total)
        for values in ([0.5, 1.5, -2.0], [], [1e300, 1e300]):
            self.assertEqual(f(values), k.list_total(values))
            self.assertEqual(f(values), k.list_total(values))
        native(self, f, 'direct')
        self.assertEqual(f([1, 2.0]), k.list_total([1, 2.0]))  # mixed types: Python path
        g = fast(k.double_list)
        a, b = [1, 2, 3], [1, 2, 3]
        g(a)
        g(a)
        k.double_list(b)
        k.double_list(b)
        self.assertEqual(a, b)
        native(self, g, 'direct')
        big, big2 = [2**62, 1], [2**62, 1]
        g(big)
        k.double_list(big2)
        self.assertEqual(big, big2)  # overflow: Python's ints, list unchanged by the native attempt

    def test_list_and_str_results(self):
        for fn, args in ((k.squares, (10,)), (k.squares, (0,)), (k.counts, (20,)), (k.shout, ('héllo', 3)), (k.shout, ('', 2))):
            f = fast(fn)
            self.assertEqual(outcome(f, *args), outcome(fn, *args))
            self.assertEqual(outcome(f, *args), outcome(fn, *args))
            native(self, f, 'direct')
        self.assertEqual(fast(k.shout)('ünï', 2), k.shout('ünï', 2))  # non-ASCII upper(): Python


class Observability(unittest.TestCase):
    def test_stats_hooks_and_logging(self):
        def loop(n):
            t = 0
            for i in range(3):
                t += n
            return t

        f = optimize(loop, warn_on_fallback=False)
        seen = []
        remove = pyrust_engine.on_fallback(lambda name, reason: seen.append((name, reason)))
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger('pyrust_engine')
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            f(2)
            f(10**30)
            f(3)
        finally:
            remove()
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        self.assertTrue(any(name.endswith('loop') and reason == 'n: int does not fit in 64 bits' for name, reason in seen))
        self.assertIn('loop runs as plain Python: n: int does not fit in 64 bits', stream.getvalue())
        rows = {r['function'].split('.')[-1]: r for r in pyrust_engine.stats() if r['function'].endswith('.loop')}
        self.assertEqual(rows['loop']['calls'], 3)
        self.assertGreaterEqual(rows['loop']['fallback_calls'], 1)
        text = io.StringIO()
        pyrust_engine.print_stats(text)
        self.assertIn('loop', text.getvalue())
        f(10**30)
        self.assertEqual(len([s for s in seen if s[1].startswith('n:')]), 1)  # removed hooks are not called
        pyrust_engine.reset_stats()
        self.assertEqual(f.report()['calls'], 0)


if __name__ == '__main__':
    unittest.main()
