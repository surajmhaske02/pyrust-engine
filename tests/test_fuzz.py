"""Generated-program differential fuzzing.

Random programs in the supported subset are written to a module, decorated, and
called natively and in CPython. Results must be identical (floats compared by
repr, so NaN and -0.0 are checked exactly) or raise the same exception type.
Every generated program must compile natively, so the fuzzer cannot silently
"pass" by falling back to Python.

Scale up locally with:
    PYRUST_FUZZ_PROGRAMS=500 PYRUST_FUZZ_SEED=7 python -m unittest tests.test_fuzz
"""
import importlib.util
import os
import random
import shutil
import tempfile
import unittest

from pyrust_engine import optimize

RUST = bool(shutil.which('rustc'))
INT_CONSTANTS = [0, 1, 2, 3, 7, 100, 2 ** 31, 2 ** 62, 2 ** 63 - 1]
INT_ARGS = [0, 1, -1, 2, -7, 13, 2 ** 40, -(2 ** 62), 2 ** 63 - 1, -(2 ** 63)]
FLOAT_CONSTANTS = ['0.0', '0.5', '1.0', '2.5', '-3.25', '1e300', '1e-300']
FLOAT_ARGS = [0.0, -0.0, 1.5, -2.25, 3.0, 1e308, -1e308, 5e-324, float('inf'), float('-inf'), float('nan')]
CMP = ['<', '<=', '>', '>=', '==', '!=']


class Generator:
    def __init__(self, rng, floats=False):
        self.rng = rng
        self.floats = floats

    # ---------------------------------------------------------------- integers
    def iexpr(self, names, loopvars, depth=0):
        rng = self.rng
        r = rng.random()
        if depth > 2 or r < 0.3:
            if names and rng.random() < 0.7:
                return rng.choice(sorted(names))
            return str(rng.choice(INT_CONSTANTS))
        if r < 0.38:
            return f'(-{self.iexpr(names, loopvars, depth + 1)})'
        op = rng.choice(['+', '-', '*', '//', '%'])
        left = self.iexpr(names, loopvars, depth + 1)
        if op == '*':  # keep CPython big-int growth linear so replays stay fast
            right = rng.choice(sorted(loopvars) + [str(c) for c in INT_CONSTANTS])
        else:
            right = self.iexpr(names, loopvars, depth + 1)
        return f'({left} {op} {right})'

    # ------------------------------------------------------------------ floats
    def fexpr(self, fnames, inames, depth=0):
        """Always float-typed: a float leaf, or a binop whose left side is float."""
        rng = self.rng
        r = rng.random()
        if depth > 2 or r < 0.3:
            if fnames and rng.random() < 0.75:
                return rng.choice(sorted(fnames))
            return rng.choice(FLOAT_CONSTANTS)
        if r < 0.36:
            return f'(-{self.fexpr(fnames, inames, depth + 1)})'
        op = rng.choice(['+', '-', '*', '/'])
        left = self.fexpr(fnames, inames, depth + 1)
        if inames and rng.random() < 0.25:
            right = rng.choice(sorted(inames))  # int operand promoted to float
        else:
            right = self.fexpr(fnames, inames, depth + 1)
        return f'({left} {op} {right})'

    def expr(self, names, loopvars):
        if self.floats:
            return self.fexpr(names, loopvars)
        return self.iexpr(names, loopvars)

    def cond(self, names, loopvars, depth=0):
        rng = self.rng
        r = rng.random()
        if depth < 1 and r < 0.2:
            joiner = rng.choice([' and ', ' or '])
            return '(' + joiner.join(self.cond(names, loopvars, depth + 1) for _ in range(2)) + ')'
        if depth < 1 and r < 0.3:
            return f'not {self.cond(names, loopvars, depth + 1)}'
        if self.floats:
            if r < 0.38:
                return self.fexpr(names, loopvars, 2)  # float truthiness
            left = self.fexpr(names, loopvars, 2)
            right = (str(rng.choice([0, 1, -5, 2 ** 53])) if rng.random() < 0.3
                     else self.fexpr(names, loopvars, 2))
            return f'{left} {rng.choice(CMP)} {right}'
        if r < 0.38:
            return self.iexpr(names, loopvars, 2)  # integer truthiness
        parts = [self.iexpr(names, loopvars, 2) for _ in range(rng.choice([2, 2, 3]))]
        out = parts[0]
        for p in parts[1:]:
            out += f' {rng.choice(CMP)} {p}'
        return out

    # -------------------------------------------------------------- statements
    def block(self, names, loopvars, indent, depth, in_loop):
        rng = self.rng
        lines = []
        pad = '    ' * indent
        pool = ['a', 'b', 'c', 't'] + sorted(n for n in names if n not in loopvars and not n.startswith('w'))
        for _ in range(rng.randint(1, 4)):
            r = rng.random()
            if r < 0.33 or depth > 2:
                tgt = rng.choice(pool)
                if tgt in names and rng.random() < 0.5:
                    ops = ['+=', '-=', '*=', '/='] if self.floats else ['+=', '-=', '//=', '%=']
                    lines.append(f'{pad}{tgt} {rng.choice(ops)} {self.expr(names, loopvars)}')
                elif rng.random() < 0.2:
                    ann = 'float' if self.floats else 'int'
                    lines.append(f'{pad}{tgt}: {ann} = {self.expr(names, loopvars)}')
                else:
                    lines.append(f'{pad}{tgt} = {self.expr(names, loopvars)}')
                names = names | {tgt}
            elif r < 0.50:
                v = rng.choice(['i', 'j', 'k'])
                form = rng.randint(1, 3)
                if form == 1:
                    rargs = f'{rng.randint(-3, 9)}'
                elif form == 2:
                    rargs = f'{rng.randint(-5, 5)}, {rng.randint(-3, 9)}'
                else:
                    rargs = f'{rng.randint(-9, 9)}, {rng.randint(-9, 9)}, {rng.choice([-3, -2, -1, 0, 1, 2, 3])}'
                lines.append(f'{pad}for {v} in range({rargs}):')
                inner_names = names if self.floats else names | {v}
                body, _ = self.block(inner_names, loopvars | {v}, indent + 1, depth + 1, True)
                lines += body
            elif r < 0.58:
                w = f'w{depth}'
                lines.append(f'{pad}{w} = 0')
                lines.append(f'{pad}while {w} < {rng.randint(0, 7)}:')
                lines.append(f'{pad}    {w} += 1')
                body, _ = self.block(names, loopvars, indent + 1, depth + 1, True)
                lines += body
            elif r < 0.82:
                lines.append(f'{pad}if {self.cond(names, loopvars)}:')
                body, n1 = self.block(names, loopvars, indent + 1, depth + 1, in_loop)
                lines += body
                if rng.random() < 0.3:
                    lines.append(f'{pad}elif {self.cond(names, loopvars)}:')
                    body, n3 = self.block(names, loopvars, indent + 1, depth + 1, in_loop)
                    lines += body
                    n1 &= n3
                lines.append(f'{pad}else:')
                body, n2 = self.block(names, loopvars, indent + 1, depth + 1, in_loop)
                lines += body
                names = names | (n1 & n2)
            elif in_loop and r < 0.90:
                lines.append(f'{pad}{rng.choice(["break", "continue"])}')
            else:
                lines.append(f'{pad}return {self.expr(names, loopvars)}')
        return lines, names

    def program(self, idx):
        if self.floats:
            params = ['x', 'y'][:self.rng.randint(1, 2)]
            sig = ', '.join(f'{p}: float' for p in params)
        else:
            params = ['x', 'y'][:self.rng.randint(0, 2)]
            sig = ', '.join(params)
        body, names = self.block(set(params), set(), 1, 0, False)
        body.append(f'    return {self.expr(names, set())}')
        return f'def f{idx}({sig}):\n' + '\n'.join(body) + '\n', len(params)


def outcome(fn, args):
    try:
        value = fn(*args)
        return ('ok', repr(value))
    except Exception as exc:  # compare exception types (ZeroDivisionError, ValueError, ...)
        return ('raise', type(exc).__name__)


@unittest.skipUnless(RUST, 'rustc required')
class GeneratedProgramFuzz(unittest.TestCase):
    def run_fuzz(self, floats, default_count, seed_offset):
        seed = int(os.environ.get('PYRUST_FUZZ_SEED', '20261106')) + seed_offset
        count = int(os.environ.get('PYRUST_FUZZ_PROGRAMS', str(default_count)))
        rng = random.Random(seed)
        gen = Generator(rng, floats=floats)
        programs = [gen.program(i) for i in range(count)]
        arg_pool = FLOAT_ARGS if floats else INT_ARGS
        with tempfile.TemporaryDirectory() as work:
            name = f'pyrust_fuzz_{"f" if floats else "i"}_{seed}'
            path = os.path.join(work, name + '.py')
            with open(path, 'w') as fh:
                fh.write('\n'.join(src for src, _ in programs))
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for i, (src, arity) in enumerate(programs):
                py = getattr(module, f'f{i}')
                fn = optimize(py, cache_dir=os.path.join(work, 'cache'))
                for _ in range(8):
                    args = [rng.choice(arg_pool) for _ in range(arity)]
                    self.assertEqual(outcome(fn, args), outcome(py, args),
                                     f'seed={seed} args={args}\n{src}\n{fn.rust_source()}')
                report = fn.report()
                self.assertEqual(report['backend'], 'rust', f'{report["compile_error"]}\n{src}')

    def test_integer_programs(self):
        self.run_fuzz(floats=False, default_count=30, seed_offset=0)

    def test_float_programs(self):
        self.run_fuzz(floats=True, default_count=20, seed_offset=1)


if __name__ == '__main__':
    unittest.main()
