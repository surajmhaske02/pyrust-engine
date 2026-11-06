import random
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pyrust_engine import OptimizationError
from tests.support import optimize

@optimize
def square_sum(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

@optimize
def arithmetic(a, b=3):
    return a * b + a - b

@optimize
def branches(n):
    total = 0
    for i in range(-5, n):
        if i < 0:
            total -= i
        else:
            total += i * i
    return total

@optimize
def loop_target(n):
    i = 20
    for i in range(n):
        i += 2
    return i

@optimize
def unsupported(n):
    return sum([n, n])

@optimize
def unbound(n):
    for i in range(n):
        x = i
    return x

@optimize
def nested(n):
    total = 0
    for i in range(n):
        for j in range(i):
            total += i * j
    return total

class Semantics(unittest.TestCase):
    def test_randomized_equivalence(self):
        rng = random.Random(17)
        for _ in range(100):
            n = rng.randint(-20, 100)
            for fn in (square_sum, branches, loop_target, nested):
                self.assertEqual(fn(n), fn.python(n))
            a, b = rng.randint(-100000, 100000), rng.randint(-100000, 100000)
            self.assertEqual(arithmetic(a, b), arithmetic.python(a, b))
    def test_big_int_and_overflow(self):
        for a, b in [(2**100, 2), (2**62, 8), (-2**63, -1), (2**63-1, 2)]:
            self.assertEqual(arithmetic(a, b), arithmetic.python(a, b))
    def test_types_keywords_and_binding(self):
        for value in (True, 2.5):
            self.assertEqual(arithmetic(value), arithmetic.python(value))
        self.assertEqual(arithmetic(b=4, a=2), 6)
        with self.assertRaises(TypeError):
            arithmetic()
        with self.assertRaises(TypeError):
            arithmetic(1, unknown=2)
    def test_unsupported_and_exception(self):
        self.assertEqual(unsupported(3), 6)
        with self.assertRaises(UnboundLocalError):
            unbound(0)
        self.assertEqual(unbound(3), 2)
    def test_range_rebinding(self):
        globals()['range'] = lambda n: [7]
        try:
            self.assertEqual(square_sum(3), 49)
        finally:
            del globals()['range']
    def test_missing_compiler(self):
        fn = optimize(arithmetic.python)
        with patch('shutil.which', return_value=None):
            self.assertEqual(fn(2), 5)
        self.assertEqual(fn.report()['backend'], 'python')
        self.assertIn('rustc', fn.report()['reason'])
    def test_strict_rejection(self):
        fn = optimize(unsupported.python, fallback=False)
        with self.assertRaises(OptimizationError):
            fn(3)
    def test_source(self):
        self.assertIn('checked_mul', square_sum.rust_source())
    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_native_and_cache(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache:
            fn = optimize(branches.python, fallback=False, cache_dir=cache)
            self.assertEqual(fn(100), branches.python(100))
            self.assertEqual(fn.report()['native_calls'], 1)
            self.assertFalse(fn.report()['cache_hit'])
            second = optimize(branches.python, fallback=False, cache_dir=cache)
            self.assertEqual(second(100), fn(100))
            self.assertTrue(second.report()['cache_hit'])
    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_native_overflow_replays(self):
        fn = optimize(arithmetic.python)
        self.assertEqual(fn(2**62, 8), arithmetic.python(2**62, 8))
        self.assertEqual(fn.report()['backend'], 'rust')
        self.assertEqual(fn.report()['fallback_calls'], 1)
    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_every_supported_kernel_is_native(self):
        for original in (square_sum, branches, loop_target, nested, arithmetic):
            fn = optimize(original.python, fallback=False)
            for n in (-3, 0, 1, 30):
                self.assertEqual(fn(n), original.python(n))
            self.assertEqual(fn.report()['native_calls'], 4)

    def test_local_range_shadow(self):
        @optimize
        def shadow(n):
            total = 0
            for i in range(n):
                total += i
            range = n
            return total
        with self.assertRaises(UnboundLocalError):
            shadow(3)
        self.assertIn('shadowing', shadow.report()['reason'])

if __name__ == '__main__':
    unittest.main()
