"""Tests for features added in 0.3.0 (merged review fixes)."""
import shutil
import threading
import unittest
from unittest.mock import patch

from tests.support import optimize
from tests import kernels as k

RUST = bool(shutil.which('rustc'))


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


def falls_off_end(n):
    if n > 0:
        return n * 2


def both_branches_return(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1


def chained(a, b, c):
    if a < b < c:
        return 1
    if a == b == c:
        return 2
    return 0


def truthiness(a, b):
    result = 0
    if a:
        result += 1
    if not b:
        result += 10
    return result


def float_truthiness(x: float):
    if x:
        return 1
    return 0


def float_vs_int_literal(x: float):
    total = 0
    if x > 0:
        total += 1
    if x == -5:
        total += 10
    if x <= 9007199254740992:
        total += 100
    return total


def unary_plus(a, x: float):
    return float(+a) + (+x)


def continue_before_assign(n):
    total = 0
    for i in range(n):
        if i % 3 == 0:
            continue
        else:
            v = i * i
        total += v
    return total


def break_in_branch(n):
    total = 0
    i = 0
    while i < n:
        i += 1
        if i > 5:
            break
        else:
            step = i
        total += step
    return total


@unittest.skipUnless(RUST, 'rustc required')
class NewSyntax(unittest.TestCase):
    def check(self, original, argsets, strict=True):
        fn = optimize(original, fallback=not strict)
        for args in argsets:
            self.assertEqual(repr(fn(*args)), repr(original(*args)), (original.__name__, args))
        self.assertEqual(fn.report()['backend'], 'rust', fn.report()['compile_error'])
        return fn

    def test_exit_aware_definite_assignment(self):
        self.check(elif_chain, [(-3,), (0,), (5,), (50,)])
        self.check(continue_before_assign, [(0,), (1,), (10,)])
        self.check(break_in_branch, [(0,), (3,), (10,)])

    def test_both_branches_return(self):
        self.check(both_branches_return, [(-7,), (0,), (8,), (9,)])

    def test_falling_off_end_returns_none_via_replay(self):
        fn = optimize(falls_off_end)
        self.assertEqual(fn(4), 8)
        self.assertIsNone(fn(-4))
        report = fn.report()
        self.assertEqual(report['backend'], 'rust')
        self.assertEqual(report['fallback_calls'], 1)
        self.assertIn('no scalar return', report['fallback_reasons'])

    def test_chained_comparisons(self):
        values = [-2, 0, 3]
        self.check(chained, [(a, b, c) for a in values for b in values for c in values])

    def test_truthiness(self):
        self.check(truthiness, [(a, b) for a in (-1, 0, 5) for b in (-1, 0, 5)])
        self.check(float_truthiness, [(0.0,), (-0.0,), (1e-300,), (float('nan'),), (float('-inf'),)])

    def test_float_compared_with_int_literal(self):
        self.check(float_vs_int_literal, [(x,) for x in
                   (0.0, -0.0, 1.5, -5.0, 2.0 ** 53, 2.0 ** 53 + 2, float('nan'), float('inf'))])

    def test_int_literal_above_2_53_is_not_converted(self):
        def too_big(x: float):
            if x < 9007199254740993:  # 2**53 + 1 is not exactly representable
                return 1
            return 0
        fn = optimize(too_big)
        self.assertEqual(fn(2.0 ** 53), 1)
        self.assertEqual(fn.report()['backend'], 'python')

    def test_unary_plus(self):
        self.check(unary_plus, [(3, 2.5), (-4, -0.0)])


class CallPath(unittest.TestCase):
    def test_positional_keyword_and_default_paths_agree(self):
        fn = optimize(k.add)
        self.assertEqual(fn(2, 3), fn(a=2, b=3))
        self.assertEqual(fn(2, b=3), 5)
        with self.assertRaises(TypeError):
            fn(1)
        with self.assertRaises(TypeError):
            fn(1, 2, 3)

    @unittest.skipUnless(RUST, 'rustc required')
    def test_counters_consistent_under_threads(self):
        fn = optimize(k.add)
        fn.prepare()
        def work():
            for i in range(500):
                fn(i, 1)
                fn(2 ** 62, 2 ** 62)  # overflow -> replay
        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        r = fn.report()
        self.assertEqual(r['calls'], 8000)
        self.assertEqual(r['native_calls'], 8000)  # every call reached native code
        self.assertEqual(r['fallback_calls'], 4000)
        self.assertEqual(r['fallback_reasons'], {'integer overflow': 4000})

    def test_unexpected_compiler_exception_falls_back(self):
        fn = optimize(k.add)
        with patch('pyrust_engine.runtime.Compiler', side_effect=KeyError('boom')):
            self.assertEqual(fn(2, 3), 5)
        self.assertEqual(fn.report()['backend'], 'python')
        self.assertIn('boom', fn.report()['compile_error'])


if __name__ == '__main__':
    unittest.main()
