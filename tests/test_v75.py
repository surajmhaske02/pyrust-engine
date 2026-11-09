"""0.7.5: methods and objects (self.x), direct calls with NumPy arrays, str and bytes."""
import copy
import dataclasses
import importlib.util
import random
import threading
import unittest

from pyrust_engine import check_rust, optimize
from pyrust_engine.compiler import Compiler, Unsupported
from tests import kernels75 as k

NUMPY = importlib.util.find_spec('numpy') is not None
RUST = check_rust()['ok']
if NUMPY:
    import numpy as np

    NUMPY2 = int(np.__version__.split('.')[0]) >= 2
else:  # pragma: no cover
    NUMPY2 = False


def fast(fn, **options):
    return optimize(fn, native_loop_free=True, warn_on_fallback=False, **options)


def state(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, '__dict__'):
        return {key: (value.tolist() if NUMPY and isinstance(value, np.ndarray) else value) for key, value in vars(obj).items()}
    return {name: getattr(obj, name, '<unset>') for name in obj.__slots__}


def run(fn, objects, *args):
    """(result or exception, object states afterwards)."""
    try:
        result = ('ok', fn(*objects, *args))
    except Exception as exc:
        result = ('raise', type(exc).__name__)
    return result, [state(o) for o in objects]


class Objects(unittest.TestCase):
    def same(self, method, make, *args, native=True):
        compiled = fast(method)
        for _ in range(3):  # compile, then cached
            a, b = make(), make()
            self.assertEqual(run(compiled, [a], *args), run(method, [b], *args), method.__qualname__)
        if RUST and native:
            report = compiled.report()
            self.assertEqual(report['backend'], 'rust', report['reason'])
            self.assertGreater(report['native_calls'], 0)
        return compiled

    def test_dataclass_method(self):
        self.same(k.Particle.advance, lambda: k.Particle(0.0, 1.0, 0.5, -2.0), 0.01, 1000)
        self.same(k.Particle.energy, lambda: k.Particle(0.0, 1.0, 0.5, -2.0), 3.0)

    def test_plain_class_and_class_attribute(self):
        self.same(k.Account.grow, lambda: k.Account(100.0, 'Ann Lee'), 30)
        self.same(k.Account.label_length, lambda: k.Account(1.0, 'Ann  Lee 🙂'))
        inactive = k.Account(100.0, 'x')
        inactive.active = False
        self.assertEqual(fast(k.Account.grow)(inactive, 5), 100.0)

    def test_slots(self):
        self.same(k.Slotted.add_up, k.Slotted, 1000)
        empty = k.Slotted.__new__(k.Slotted)  # attributes never set: AttributeError like Python
        self.assertEqual(run(fast(k.Slotted.add_up), [empty], 3), run(k.Slotted.add_up, [k.Slotted.__new__(k.Slotted)], 3))

    def test_numpy_attribute(self):
        if not NUMPY:
            self.skipTest('NumPy required')
        self.same(k.Grid.double_all, lambda: k.Grid(np.arange(10, dtype=np.float64)))
        self.same(k.Grid.double_all, lambda: k.Grid([1.0, 2.0]), native=False)  # list attribute: Python

    def test_runs_as_python_with_same_result(self):
        cases = [
            (k.WithProperty.triple, k.WithProperty, (4,)),  # property: Python
            (k.Frozen.bump, lambda: k.Frozen(1), (3,)),  # FrozenInstanceError like Python
            (k.Counter.blow_up, lambda: k.Counter(7), (10,)),  # overflow: replay, object untouched first
            (k.Counter.to_float, lambda: k.Counter(1), ()),  # int attribute becomes float
            (k.Particle.calls_method, lambda: k.Particle(0.0, 1.0, 0.5, -2.0), ()),  # method call: Python
            (k.Counter.blow_up, lambda: k.Counter(2**70), (1,)),  # big int attribute
        ]
        for method, make, args in cases:
            self.same(method, make, *args, native=False)
        missing = k.Account.__new__(k.Account)
        self.assertEqual(run(fast(k.Account.grow), [missing], 1)[0], ('raise', 'AttributeError'))

    def test_two_objects_and_aliasing(self):
        compiled = fast(k.transfer)
        a, b, c, d = k.Account(10.0, 'a'), k.Account(5.0, 'b'), k.Account(10.0, 'a'), k.Account(5.0, 'b')
        self.assertEqual(run(compiled, [a, b], 1.5), run(k.transfer, [c, d], 1.5))
        same, same2 = k.Account(10.0, 'a'), k.Account(10.0, 'a')
        self.assertEqual(run(compiled, [same, same], 1.5)[0], run(k.transfer, [same2, same2], 1.5)[0])
        self.assertEqual(same.balance, same2.balance)

    def test_decorated_in_class(self):
        a, b = k.Decorated(), k.Decorated()
        for n in (1, 5, 1000):
            self.assertEqual(a.hit(n), k.Decorated.hit.python(b, n))
        self.assertEqual(a.hits, b.hits)

    def test_compiler_rules(self):
        kinds = {'self': 'obj(x:f64,vx:f64)', 'dt': 'f64', 'n': 'i64'}
        with self.assertRaises(Unsupported) as caught:
            Compiler(k.Particle.calls_method, kinds={'self': 'obj()'}).generate()
        self.assertIn('methods are compiled only when the object comes from a real call', str(caught.exception))
        source = Compiler(
            k.Particle.advance, kinds={**kinds, 'self': 'obj(steps:i64!,vx:f64,vy:f64!,x:f64!,y:f64!)'}
        ).generate()
        self.assertIn('fn compute', source)
        self.assertNotIn('pyrust_py_call', source)  # objects use the normal path


# --------------------------------------------------------------- direct calls
def direct(test, fn):
    if RUST:
        test.assertEqual(fn.report()['dispatch'], 'direct', fn.report()['reason'])


def outcome(fn, *args):
    try:
        return ('ok', fn(*args))
    except Exception as exc:
        return ('raise', type(exc).__name__)


@unittest.skipUnless(NUMPY, 'NumPy required')
class DirectArrays(unittest.TestCase):
    def test_reads(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(1000)
        f = fast(k.total)
        for values in (a, a[:0], a[:1], np.zeros(5)):
            self.assertEqual(f(values), k.total(values))
            self.assertEqual(f(values), k.total(values))
        direct(self, f)
        ints = rng.integers(-1000, 1000, 500)
        g = fast(k.int_total)
        self.assertEqual(g(ints), k.int_total(ints))
        self.assertEqual(g(ints), k.int_total(ints))
        direct(self, g)
        big = np.array([2**62, 2**62], dtype=np.int64)
        self.assertEqual(g(big), k.int_total(big))  # overflow: Python's int result
        m = rng.standard_normal((7, 5))
        h = fast(k.grid_sum)
        self.assertEqual(h(m), k.grid_sum(m))
        self.assertEqual(h(m), k.grid_sum(m))
        direct(self, h)
        if NUMPY2:
            a32 = a.astype(np.float32)
            f32 = fast(k.total)
            self.assertEqual(outcome(f32, a32), outcome(k.total, a32))
            self.assertEqual(outcome(f32, a32), outcome(k.total, a32))

    def test_other_arrays_take_the_python_path(self):
        a = np.arange(20, dtype=np.float64)
        f = fast(k.total)
        f(a)
        for values in (a[::2], a.reshape(4, 5).T.copy().ravel()[::3], a.astype('>f8'), np.asfortranarray(a.reshape(4, 5))[:, 0]):
            self.assertEqual(outcome(f, values), outcome(k.total, values))
        sub = a.view(type('Sub', (np.ndarray,), {}))
        self.assertEqual(f(sub), k.total(sub))
        self.assertEqual(f([1.0, 2.0]), 3.0)
        self.assertEqual(f(np.arange(5)), k.total(np.arange(5)))  # int64 array: another specialization

    def test_writes(self):
        a = np.arange(1, 9, dtype=np.float64)
        out, expected_out = np.zeros(8), np.zeros(8)
        f = fast(k.scaled_into)
        for _ in range(3):
            self.assertEqual(f(a, out, 2.5), k.scaled_into(a, expected_out, 2.5))
            self.assertEqual(out.tolist(), expected_out.tolist())
        direct(self, f)
        # the same array as input and output: Python path (writes would overlap)
        b, c = a.copy(), a.copy()
        self.assertEqual(f(b, b, 2.0), k.scaled_into(c, c, 2.0))
        self.assertEqual(b.tolist(), c.tolist())
        frozen = np.zeros(8)
        frozen.flags.writeable = False
        self.assertEqual(outcome(f, a, frozen, 2.0), outcome(k.scaled_into, a, frozen, 2.0))  # ValueError
        g = fast(k.fill_until)
        x, y = np.full(6, 7.0), np.full(6, 7.0)
        self.assertEqual(outcome(g, x, 3), outcome(k.fill_until, y, 3))
        self.assertEqual(x.tolist(), y.tolist())  # Python's partial writes, after a native undo

    def test_threads(self):
        a = np.random.default_rng(4).standard_normal(300)
        f = fast(k.total)
        f(a)
        errors = []

        def work():
            for _ in range(2000):
                if f(a) != k.total(a):
                    errors.append(1)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_mixed_arguments(self):
        a = np.arange(10, dtype=np.float64)
        f = fast(k.mixed)
        for args in ((a, 'héllo 🙂', 5, 0.5), (a, '', 20, 2.0), (a, 'abc', 0, 1.0)):
            self.assertEqual(f(*args), k.mixed(*args))
            self.assertEqual(f(*args), k.mixed(*args))
        direct(self, f)


class DirectText(unittest.TestCase):
    TEXTS = ['', 'a', 'education', 'rhythm', 'ünïcode vowels aeiou', '🙂 aei 🙂', 'x' * 10000, '\udcff lone surrogate']

    def test_str(self):
        f = fast(k.vowels)
        for text in self.TEXTS:
            self.assertEqual(f(text), k.vowels(text))
            self.assertEqual(f(text), k.vowels(text))
        direct(self, f)
        g = fast(k.starts_with)
        for text in self.TEXTS:
            for prefix in ('', 'e', 'ü', 'xx'):
                self.assertEqual(g(text, prefix), k.starts_with(text, prefix))
        direct(self, g)
        self.assertEqual(outcome(f, 5), outcome(k.vowels, 5))
        self.assertEqual(f(type('S', (str,), {})('aaa')), 3)  # str subclass: Python path

    def test_bytes(self):
        f = fast(k.checksum)
        rng = random.Random(5)
        for data in (b'', b'abc', bytes(rng.randrange(256) for _ in range(5000))):
            self.assertEqual(f(data), k.checksum(data))
            self.assertEqual(f(data), k.checksum(data))
        direct(self, f)
        self.assertEqual(f(bytearray(b'abc')), k.checksum(bytearray(b'abc')))  # bytearray: Python path


if __name__ == '__main__':
    unittest.main()
