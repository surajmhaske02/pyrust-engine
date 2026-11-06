import random
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pyrust_engine import optimize, OptimizationError

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

@optimize
def elif_chain(n):
    if n < 0:
        return -1
    elif n == 0:
        return 0
    elif n < 10:
        r = n * 2
    else:
        r = n
    return r

@optimize
def annotated(n: int) -> int:
    total: int = 0
    for i in range(n):
        total += i
    return total

@optimize
def divmod_kernel(a, b):
    return a // b * 1000 + a % b

@optimize
def stepped(start, stop, step):
    total = 0
    for i in range(start, stop, step):
        total = total * 3 + i
    return total

@optimize
def count_range(start, stop, step):
    count = 0
    last = 0
    for i in range(start, stop, step):
        count += 1
        last = i
    return count * 1000 + last % 1000

@optimize
def logic(a, b):
    if a and not b:
        return 1
    if 0 <= a < b or a == -b:
        return 2
    return 3

class NewFeatures(unittest.TestCase):
    """Features added after the v0.1 review."""
    def check_native(self, original, argsets):
        fn = optimize(original.python, fallback=False)
        for args in argsets:
            self.assertEqual(fn(*args), original.python(*args), args)
        self.assertEqual(fn.report()['backend'], 'rust')

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_return_aware_definite_assignment(self):
        self.check_native(elif_chain, [(-3,), (0,), (5,), (50,)])

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_annotated_assignment(self):
        self.check_native(annotated, [(0,), (10,)])

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_floor_division_and_modulo_follow_python(self):
        values = [-7, -3, -1, 1, 3, 7, 2**63 - 1, -2**63]
        self.check_native(divmod_kernel, [(a, b) for a in values for b in values
                                          if not (a == -2**63 and b == -1) and abs(a // b * 1000) < 2**62
                                          and abs(a // b * 1000 + a % b) < 2**62])
        fn = optimize(divmod_kernel.python)
        with self.assertRaises(ZeroDivisionError):
            fn(5, 0)
        self.assertEqual(fn(-2**63, -1), divmod_kernel.python(-2**63, -1))  # overflow -> replay

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_range_step(self):
        cases = [(0, 10, 3), (10, 0, -3), (5, 5, 1), (-4, 4, 2), (9, -9, -4), (0, 3, 7)]
        self.check_native(stepped, cases)
        # index + step overflows i64 after the last element: loop must just end.
        self.check_native(count_range, [(2**63 - 10, 2**63 - 1, 4), (-2**63 + 10, -2**63, -4),
                                        (0, 2**63 - 1, 2**62), (3, -3, -1)])
        with self.assertRaises(ValueError):
            optimize(stepped.python)(0, 5, 0)

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_boolean_conditions_and_chained_compare(self):
        self.check_native(logic, [(a, b) for a in range(-3, 4) for b in range(-3, 4)])

    def test_keyword_and_positional_paths_agree(self):
        self.assertEqual(arithmetic(2, 4), arithmetic(a=2, b=4))
        self.assertEqual(arithmetic(2), arithmetic(2, 3))

    @unittest.skipUnless(shutil.which('rustc'), 'rustc required')
    def test_concurrent_first_call(self):
        import threading
        with tempfile.TemporaryDirectory() as cache:
            fn = optimize(square_sum.python, cache_dir=cache)
            results, errors = [], []
            def worker():
                try:
                    results.append(fn(1000))
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(results, [square_sum.python(1000)] * 8)
            self.assertEqual(fn.report()['backend'], 'rust')

    def test_internal_errors_fall_back(self):
        from unittest.mock import patch as _patch
        fn = optimize(arithmetic.python)
        with _patch('pyrust_engine.runtime.Compiler', side_effect=RuntimeError('boom')):
            self.assertEqual(fn(2, 3), 5)
        self.assertIn('Internal compiler error', fn.report()['reason'])

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
        with tempfile.TemporaryDirectory() as cache:
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
