"""0.7.2: float32, string values, growable lists, dicts and sets."""
import importlib.util
import random
import unittest

from pyrust_engine import check_rust, optimize
from pyrust_engine.compiler import Compiler, Unsupported
from pyrust_engine.packing import pack, unpack

NUMPY = importlib.util.find_spec('numpy') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    NUMPY2 = int(np.__version__.split('.')[0]) >= 2
    from tests import kernels72 as k
else:  # pragma: no cover
    NUMPY2 = False


def normal(value):
    """Comparable form: sets have no reproducible order, NumPy values compare by bytes."""
    if isinstance(value, set):
        return ('set', sorted(value, key=repr))
    if NUMPY and isinstance(value, np.ndarray):
        return ('array', str(value.dtype), value.tobytes())
    if isinstance(value, list):
        return ('list', [normal(v) for v in value])
    if isinstance(value, dict):
        return ('dict', [(repr(a), normal(b)) for a, b in value.items()])
    return (type(value).__name__, repr(value))


def outcome(fn, args):
    try:
        return ('ok', normal(fn(*args)))
    except Exception as exc:
        return ('raise', type(exc).__name__)


def fast(fn):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False)


TEXTS = [
    'The quick brown fox jumps over the lazy dog and the cat',
    '',
    '   spaces   everywhere   ',
    'one',
    'Numbers 1 2 3 and words, punctuation! The END.',
    'tabs\tand\nnewlines\x0bvertical em space',
    'emoji 🙂 and ünïcode wörds',
]


class Packing(unittest.TestCase):
    def test_round_trip(self):
        for kind, value in (
            ('slist', ['a', '', 'héllo', '🙂🙂', 'x' * 1000]),
            ('slist', []),
            ('set(strv)', {'a', 'b', ''}),
            ('set(i64)', {1, -5, 2**62}),
            ('dict(strv,f64)', {'a': 1.5, 'b': -0.0, 'c': float('inf')}),
            ('dict(i64,strv)', {3: 'x', 1: 'yy', -2: ''}),
            ('dict(strv,bool)', {'t': True, 'f': False}),
            ('dict(strv,i64)', {}),
        ):
            self.assertEqual(unpack(kind, pack(kind, value)), value)
        raw = 'one 🙂'.encode('utf-32-le')
        header = (2**64 - 1).to_bytes(8, 'little') + len('one 🙂').to_bytes(8, 'little') * 2
        self.assertEqual(unpack('strv', header + raw), 'one 🙂')
        joined = '\x00'.join(['ab', '', 'c']).encode('utf-32-le')
        self.assertEqual(unpack('slist', (3).to_bytes(8, 'little') + bytes(8) + (5).to_bytes(8, 'little') + joined), ['ab', '', 'c'])


@unittest.skipUnless(NUMPY, 'NumPy required')
class Compile(unittest.TestCase):
    """Pure-Python checks: run with or without Rust."""

    def test_kernels_compile(self):
        expected = {
            'dot32': 'f64', 'scale32': 'farray', 'row_scores': 'none', 'mean32': 'f32', 'clip32': 'i64',
            'words_upper': 'slist', 'tokenize': 'slist', 'chunk': 'slist', 'string_tools': 'i64',
            'join_words': 'strv', 'parse_numbers': 'f64', 'first_word': 'strv', 'char_list': 'slist',
            'greet': 'strv', 'squares': 'ilist', 'evens_then_sort': 'flist', 'sorted_words': 'slist',
            'list_ops': 'i64', 'build_matrix_row': 'flist', 'word_counts': 'dict(strv,i64)', 'top_word': 'strv',
            'bm25_like': 'f64', 'jaccard': 'f64', 'dedupe': 'slist', 'dict_loops': 'i64',
            'grow_while_looping': 'i64', 'int_keys': 'dict(i64,f64)', 'set_ops': 'set(strv)', 'letters': 'i64',
            'lookup': 'i64', 'uses_helper': 'dict(strv,i64)',
        }
        for name, kind in expected.items():
            if name in ('dot32', 'scale32', 'row_scores', 'mean32', 'clip32') and not NUMPY2:
                continue
            compiler = Compiler(getattr(k, name))
            compiler.generate()
            self.assertEqual(compiler.return_type, kind, name)

    def test_rules_with_hints(self):
        def alias(values: list[float]) -> float:
            other = values
            return other[0]

        def mixed_list(n: int) -> list[float]:
            out = []
            for i in range(n):
                out.append(i)
                out.append(0.5)
            return out

        def float_keys(n: int) -> int:
            d = {}
            for i in range(n):
                d[i * 0.5] = i
            return len(d)

        def set_loop(words: list[str]) -> int:
            total = 0
            for w in set(words):
                total += len(w)
            return total

        def numpy_slice(a: k.F32Array) -> float:
            total = 0.0
            for x in a[1:]:
                total += float(x)
            return total

        def f32_strong(a: k.F32Array, b: k.F32Array, c) -> float:
            total = 0.0
            for i in range(len(a)):
                total += float(a[i] * c[i])
            return total

        def float_fstring(x: float) -> str:
            return f'{x}'

        for fn, kinds, text in (
            (alias, None, 'aliased'),
            (mixed_list, None, 'holds'),
            (float_keys, None, 'keys'),
            (set_loop, None, 'sorted(s)'),
            (numpy_slice, {'a': 'farray'}, 'views'),
            (f32_strong, {'a': 'farray', 'b': 'farray', 'c': 'array'}, 'float32'),
            (float_fstring, None, 'float formatting'),
        ):
            with self.assertRaises(Unsupported, msg=fn.__name__) as ctx:
                Compiler(fn, kinds=kinds).generate()
            self.assertIn(text, str(ctx.exception), fn.__name__)

    def test_read_only_arguments(self):
        with self.assertRaises(Unsupported) as ctx:
            Compiler(k.helper_counts).generate()
        self.assertIn('read-only', str(ctx.exception))

    def test_python_results_without_rust(self):
        self.assertEqual(fast(k.word_counts)('a b a'), k.word_counts('a b a'))
        self.assertEqual(fast(k.tokenize)('Hi, you!'), ['hi', 'you'])


@unittest.skipUnless(RUST and NUMPY, 'Rust toolchain and NumPy required')
class Native(unittest.TestCase):
    def assert_same(self, fn, *cases, native=True):
        f = fast(fn)
        for args in cases:
            self.assertEqual(outcome(f, args), outcome(fn, args), (fn.__name__, args))
        if native:
            self.assertTrue(f.report()['specializations'], (fn.__name__, f.report()['reason']))
        return f

    @unittest.skipUnless(NUMPY2, 'float32 needs NumPy 2')
    def test_float32(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal(257).astype(np.float32)
        b = rng.standard_normal(257).astype(np.float32)
        self.assert_same(k.dot32, (a, b), (a[:0], b[:0]))
        self.assert_same(k.dot32_in_f64, (a, b))
        self.assert_same(k.scale32, (a, 0.1), (a, 3))
        self.assert_same(k.mean32, (a,), (a[:0],))
        self.assert_same(k.clip32, (a, 1.0))
        m = rng.standard_normal((40, 17)).astype(np.float32)
        q = rng.standard_normal(17).astype(np.float32)
        out_fast, out_slow = np.zeros(40, dtype=np.float32), np.zeros(40, dtype=np.float32)
        fast(k.row_scores)(m, q, out_fast)
        k.row_scores(m, q, out_slow)
        self.assertEqual(out_fast.tobytes(), out_slow.tobytes())
        self.assertIsInstance(fast(k.mean32)(a), np.float32)

    def test_strings(self):
        for fn in (k.words_upper, k.tokenize, k.first_word, k.char_list):
            self.assert_same(fn, *[(t,) for t in TEXTS], native=False)
        self.assert_same(k.chunk, *[(t * 3, 20, 5) for t in TEXTS], (TEXTS[0], 7, 0))
        self.assert_same(k.string_tools, ('the cat and the hat', 'the'), ('abc', 'zz'), ('the', 'the'))
        self.assert_same(k.join_words, (TEXTS[0].split(),), ([],), (['a', 'bb', 'the'],))
        self.assert_same(k.parse_numbers, ('1.5, 2, -3e2 , ',), (' 7 ',), ('1_0',), ('nan, 1',), ('x',))
        self.assert_same(k.greet, ('World',), ('🙂',))
        self.assert_same(k.tokenize, ('Hello, World! 42 times',))
        self.assert_same(k.words_upper, ('only ascii words here',))

    def test_lists(self):
        self.assert_same(k.squares, (0,), (10,), (1000,))
        self.assert_same(k.evens_then_sort, ([3.0, -1.0, 2.5, 2.5, 0.0, 10.0],), ([],), ([-1.0],))
        self.assert_same(k.sorted_words, (TEXTS[0],), ('',), ('b a c a',))
        self.assert_same(k.list_ops, ([5, 3, 9, 3, 1],), ([7, 7, 7],))
        self.assert_same(k.build_matrix_row, (0,), (5,))
        self.assert_same(k.list_ops, ([1],), native=False)  # index of a missing value: ValueError in both

    def test_dicts_and_sets(self):
        for text in TEXTS:
            self.assert_same(k.word_counts, (text,), native=False)
            self.assert_same(k.letters, (text,))
            self.assert_same(k.jaccard, (text, TEXTS[0]))
        self.assert_same(k.word_counts, (TEXTS[0],))
        self.assert_same(k.top_word, ('b a b c a b',), ('x',), ('',))
        idf = {'fox': 1.5, 'dog': 0.7, 'cat': 2.0}
        self.assert_same(k.bm25_like, (['fox', 'cat', 'zzz'], TEXTS[0].split(), idf), ([], [], {}))
        self.assert_same(k.dedupe, (['b', 'a', 'b', 'c', 'a'],), ([],))
        self.assert_same(k.dict_loops, ('hello world',), ('s',))
        self.assert_same(k.grow_while_looping, (1,), (5,))
        self.assert_same(k.int_keys, ([1, 2, 6, 11, -3],))
        self.assert_same(k.set_ops, (['a', 'b', 'x'], ['b', 'c']))
        self.assert_same(k.lookup, ({'a': 1, 'b': 2}, ['a', 'c', 'b', 'a']))
        self.assert_same(k.uses_helper, (TEXTS[0],))
        f = fast(k.word_counts)
        self.assertEqual(list(f(TEXTS[0])), list(k.word_counts(TEXTS[0])))  # same insertion order
        self.assertEqual(f.report()['backend'], 'rust')

    def test_random_still_exact_with_lists(self):
        def shuffle_pick(words: list[str], n: int) -> list[str]:
            out = []
            for _ in range(n):
                out.append(words[random.randrange(len(words))])
            return out

        f = fast(shuffle_pick)
        random.seed(3)
        expected = shuffle_pick(['a', 'b', 'c'], 50)
        random.seed(3)
        self.assertEqual(f(['a', 'b', 'c'], 50), expected)


if __name__ == '__main__':
    unittest.main()
