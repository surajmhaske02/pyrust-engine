"""0.6.0: recursion, array writes (copy-on-write), I64Array, None returns,
exact int/float comparisons and type information."""
import importlib.util
import shutil
import unittest

from pyrust_engine import optimize, OptimizationError, I64Array, F64Array
from pyrust_engine.compiler import Compiler, Unsupported
from tests import kernels6 as k

RUST = bool(shutil.which('rustc'))
NUMPY = importlib.util.find_spec('numpy') is not None


def outcome(fn, args):
    try:
        return ('ok', repr(fn(*args)))
    except Exception as exc:
        return ('raise', type(exc).__name__)


class Compile(unittest.TestCase):
    """Pure-Python checks: run with or without Rust."""

    def test_new_kernels_compile(self):
        for name in ('fib', 'is_even', 'depth', 'power_sum', 'scale', 'prefix_sum', 'normalize',
                     'accumulate', 'histogram', 'int_sum', 'int_to_float', 'write_then_fail',
                     'copy_shift', 'count_below', 'compare_pairs'):
            Compiler(getattr(k, name)).generate()

    def test_written_arrays_detected(self):
        compiler = Compiler(k.prefix_sum)
        compiler.generate()
        self.assertEqual(compiler.written, {'out'})
        self.assertEqual(compiler.return_type, 'f64')
        none = Compiler(k.scale)
        none.generate()
        self.assertEqual(none.return_type, 'none')

    def test_int_array_rejects_float_write(self):
        def bad(a: I64Array):
            for i in range(len(a)):
                a[i] = 1.5
        with self.assertRaises(Unsupported) as ctx:
            Compiler(bad).generate()
        self.assertIn('int arrays can only be set to int values', str(ctx.exception))

    def test_callee_can_write(self):
        # 0.7: helpers may write arrays they receive; the caller's array becomes written too.
        compiler = Compiler(_calls_writer)
        compiler.generate()
        self.assertEqual(compiler.written, {'a'})

    def test_explain_mentions_types(self):
        text = optimize(k.histogram, warn_on_fallback=False).explain()
        self.assertIn('data: I64Array', text)
        self.assertIn('returns: None', text)

    def test_typing_protocol_exported(self):
        from pyrust_engine import OptimizedFunction
        fn = optimize(k.int_sum, warn_on_fallback=False)
        self.assertTrue(callable(fn.explain))
        self.assertTrue(hasattr(OptimizedFunction, '__call__'))


def _writer(a: F64Array, i: int) -> int:
    a[i] = 1.0
    return i


def _calls_writer(a: F64Array) -> int:
    total = 0
    for i in range(len(a)):
        total += _writer(a, i)
    return total


@unittest.skipUnless(RUST, 'rustc required')
class Recursion(unittest.TestCase):
    def test_fib(self):
        fn = optimize(k.fib, fallback=False)
        for n in (0, 1, 2, 10, 20, 25):
            self.assertEqual(fn(n), k.fib(n))
        self.assertEqual(fn.report()['backend'], 'rust')

    def test_mutual_recursion(self):
        fn = optimize(k.is_even, fallback=False)
        for n in (0, 1, 7, 100, 251):
            self.assertEqual(fn(n), k.is_even(n))

    def test_deep_recursion_replays(self):
        fn = optimize(k.depth, warn_on_fallback=False)
        self.assertEqual(fn(100), 100)
        self.assertEqual(fn.report()['fallback_calls'], 0)
        self.assertEqual(fn(800), 800)            # deeper than the native limit: replayed
        self.assertEqual(fn.report()['fallback_calls'], 1)
        self.assertEqual(outcome(fn, (5000,)), outcome(k.depth, (5000,)))  # RecursionError in both

    def test_recursion_budget(self):
        fn = optimize(k.fib, max_iterations=1000, warn_on_fallback=False)
        self.assertEqual(fn(25), k.fib(25))
        self.assertIn('budget', fn.report()['reason'])

    def test_float_recursion(self):
        fn = optimize(k.power_sum, warn_on_fallback=False)
        for args in ((0, 2.0), (10, 1.5), (5, -0.5), (3, float('nan')), (2000, 1.0001)):
            self.assertEqual(outcome(fn, args), outcome(k.power_sum, args))


@unittest.skipUnless(RUST and NUMPY, 'rustc and NumPy required')
class ArrayWrites(unittest.TestCase):
    def same_effect(self, kernel, make_args, *, native=True):
        """Run the kernel natively and in Python on separate copies; compare
        the return value (or exception) and every array afterwards."""
        import numpy as np
        fn = optimize(kernel, warn_on_fallback=False)
        native_args = make_args()
        python_args = make_args()
        self.assertEqual(outcome(fn, native_args), outcome(kernel, python_args))
        for a, b in zip(native_args, python_args):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.tobytes(), b.tobytes())  # exact bits, including -0.0 and NaN
        if native:
            self.assertEqual(fn.report()['backend'], 'rust', fn.report()['compile_error'])
            self.assertGreater(fn.report()['native_calls'], 0)
        return fn

    def test_scale_returns_none(self):
        import numpy as np
        rng = np.random.default_rng(1)
        values = rng.standard_normal(1000)
        fn = self.same_effect(k.scale, lambda: (values.copy(), np.zeros(1000), 2.5))
        self.assertEqual(fn.report()['fallback_calls'], 0)

    def test_prefix_sum(self):
        import numpy as np
        values = np.arange(100, dtype=np.float64) * 0.1
        self.same_effect(k.prefix_sum, lambda: (values.copy(), np.empty(100)))
        self.same_effect(k.prefix_sum, lambda: (values.copy(), np.empty(10)))   # IndexError

    def test_normalize_and_zero_total(self):
        import numpy as np
        import warnings
        self.same_effect(k.normalize, lambda: (np.array([1.0, 2.0, 5.0]),))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # NumPy warns on 0/0 in the Python replay
            self.same_effect(k.normalize, lambda: (np.array([1.0, -1.0, 0.0]),))

    def test_accumulate_negative_index(self):
        import numpy as np
        self.same_effect(k.accumulate, lambda: (np.zeros(7), 50))
        self.same_effect(k.accumulate, lambda: (np.zeros(0), 3))   # ZeroDivisionError

    def test_failure_leaves_array_like_python(self):
        import numpy as np
        fn = self.same_effect(k.write_then_fail, lambda: (np.zeros(10), 10))
        self.assertEqual(fn.report()['fallback_calls'], 1)
        self.same_effect(k.write_then_fail, lambda: (np.zeros(10), 4))

    def test_overlapping_arrays_fall_back(self):
        import numpy as np
        fn = optimize(k.copy_shift, warn_on_fallback=False)
        shared = np.arange(6, dtype=np.float64)
        expected = shared.copy()
        k.copy_shift(expected, expected)
        fn(shared, shared)
        np.testing.assert_array_equal(shared, expected)
        self.assertIn('share memory', fn.report()['reason'])
        base = np.arange(10, dtype=np.float64)
        fn(base[:6], base[3:9])       # overlapping views
        check = np.arange(10, dtype=np.float64)
        k.copy_shift(check[:6], check[3:9])
        np.testing.assert_array_equal(base, check)
        a, b = np.arange(5, dtype=np.float64), np.zeros(5)
        fn(a, b)                      # separate arrays: native
        self.assertEqual(list(b), [0.0, 0.0, 1.0, 2.0, 3.0])

    def test_read_only_array_raises_like_python(self):
        import numpy as np
        out = np.zeros(3)
        out.setflags(write=False)
        fn = optimize(k.scale, warn_on_fallback=False)
        with self.assertRaises(ValueError):
            fn(np.ones(3), out, 2.0)

    def test_int64_arrays(self):
        import numpy as np
        data = np.arange(-50, 1000, dtype=np.int64)
        self.same_effect(k.histogram, lambda: (data.copy(), np.zeros(7, dtype=np.int64), 7))
        self.same_effect(k.histogram, lambda: (data.copy(), np.zeros(7, dtype=np.int64), 0))  # ZeroDivisionError
        self.same_effect(k.int_sum, lambda: (data.copy(),))
        self.same_effect(k.int_to_float, lambda: (data.copy(), np.zeros(data.size)))

    def test_int_sum_overflow_replays(self):
        import numpy as np
        big = np.array([2**62, 2**62, 2**62], dtype=np.int64)
        fn = self.same_effect(k.int_sum, lambda: (big.copy(),))
        self.assertEqual(fn(big), 3 * 2**62)

    def test_wrong_dtype_falls_back(self):
        import numpy as np
        fn = optimize(k.int_sum, warn_on_fallback=False)
        self.assertEqual(fn(np.arange(5, dtype=np.int32)), 10)
        self.assertIn('int64', fn.report()['reason'])


@unittest.skipUnless(RUST, 'rustc required')
class MixedComparisons(unittest.TestCase):
    def test_exact_int_float_comparisons(self):
        # 0.7: ints beyond 2**53 replay in Python (NumPy scalars compare them differently), so
        # results are still exact but those calls are not native.
        fn = optimize(k.compare_pairs, warn_on_fallback=False)
        ints = [0, 1, -1, 3, 2**53, 2**53 + 1, 2**62, 2**63 - 1, -(2**63)]
        floats = [0.0, -0.0, 0.5, -0.5, 3.0, 2.9999999999999996, 9007199254740992.0, 9007199254740994.0,
                  4.611686018427388e18, 9.223372036854776e18, -9.223372036854776e18,
                  float('inf'), float('-inf'), float('nan')]
        for i in ints:
            for f in floats:
                self.assertEqual(fn(i, f), k.compare_pairs(i, f), (i, f))

    def test_count_below(self):
        fn = optimize(k.count_below, fallback=False)
        for limit in (0.0, 2.5, 10.0, -1.0, float('nan'), float('inf')):
            self.assertEqual(fn(limit, 20), k.count_below(limit, 20))


if __name__ == '__main__':
    unittest.main()
