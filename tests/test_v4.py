"""0.4.0: loop-free functions run as plain Python with zero dispatch overhead."""
import shutil
import timeit
import types
import unittest

from pyrust_engine import optimize
from tests import kernels as k

RUST = bool(shutil.which('rustc'))


def keyword_defaults(a, b=2, *, c=3):
    return a * b + c


def with_loop(n):
    total = 0
    for i in range(n):
        total += i
    return total


def with_while(n):
    while n > 10:
        n -= 3
    return n


class LoopFree(unittest.TestCase):
    def test_loop_free_runs_as_plain_function(self):
        fn = optimize(k.add)
        self.assertIsInstance(fn, types.FunctionType)
        self.assertIsNot(fn, k.add)
        self.assertIs(fn.python, k.add)
        self.assertEqual(fn(2, 3), 5)
        self.assertEqual(fn(2 ** 70, 1), 2 ** 70 + 1)  # big ints stay exact
        self.assertEqual(fn('a', 'b'), 'ab')            # any type works, it is Python
        self.assertEqual(fn.__name__, 'add')
        report = fn.report()
        self.assertEqual(report['backend'], 'python')
        self.assertEqual(report['dispatch'], 'python-direct')
        self.assertFalse(report['call_counting'])
        self.assertIn('loop-free', report['reason'])
        self.assertEqual(fn.prepare()['backend'], 'python')
        self.assertEqual(fn.reset()['backend'], 'python')

    def test_all_loop_free_kernels(self):
        for kernel, args in [(k.divmod_mix, (17, 5)), (k.float_math, (1.5, 2.0)), (k.bool_logic, (1, 11)),
                             (k.min_literal, (5,)), (k.add, (-3, 9))]:
            fn = optimize(kernel)
            self.assertEqual(fn.report()['dispatch'], 'python-direct', kernel.__name__)
            self.assertEqual(fn(*args), kernel(*args))

    def test_exceptions_match_python(self):
        fn = optimize(k.divmod_mix)
        with self.assertRaises(ZeroDivisionError):
            fn(1, 0)

    def test_defaults_and_keywords(self):
        fn = optimize(keyword_defaults)
        self.assertEqual(fn(4), 11)
        self.assertEqual(fn(4, 3, c=1), 13)
        self.assertEqual(fn(a=1, b=1), 4)
        with self.assertRaises(TypeError):
            fn()

    def test_decorator_forms(self):
        @optimize
        def bare(x):
            return x + 1

        @optimize(fallback=False)
        def configured(x):
            return x * 2

        self.assertEqual(bare(1), 2)
        self.assertEqual(configured(4), 8)
        self.assertEqual(bare.report()['dispatch'], 'python-direct')
        self.assertEqual(configured.report()['dispatch'], 'python-direct')

    def test_functions_with_loops_keep_native_wrapper(self):
        for kernel in (with_loop, with_while, k.stepped, k.sum_squares):
            fn = optimize(kernel)
            self.assertNotEqual(fn.report()['dispatch'], 'python-direct', kernel.__name__)
            self.assertTrue(fn.report()['call_counting'])

    def test_rust_source_still_available(self):
        self.assertIn('pyrust_run', optimize(k.add).rust_source())

    def test_no_wrapper_overhead(self):
        fn = optimize(k.add)
        plain = min(timeit.repeat(lambda: k.add(1, 2), number=20000, repeat=5))
        fast = min(timeit.repeat(lambda: fn(1, 2), number=20000, repeat=5))
        # Same bytecode, so timing should match; allow generous noise for shared CI machines.
        self.assertLess(fast, plain * 3)

    @unittest.skipUnless(RUST, 'rustc required')
    def test_force_native(self):
        fn = optimize(k.add, native_loop_free=True, fallback=False)
        self.assertEqual(fn(2, 3), 5)
        self.assertEqual(fn.report()['backend'], 'rust')
        self.assertEqual(fn.report()['native_calls'], 1)


if __name__ == '__main__':
    unittest.main()
