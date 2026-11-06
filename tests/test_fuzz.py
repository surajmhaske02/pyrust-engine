"""Differential fuzzing: random programs in the supported subset must give the
same result (or raise the same exception type) natively and in CPython.

Run more programs with:  PYRUST_FUZZ_PROGRAMS=500 PYRUST_FUZZ_SEED=7 python -m unittest tests.test_fuzz
"""
import importlib.util
import os
import random
import shutil
import tempfile
import unittest

from pyrust_engine import optimize

CONSTANTS = [0, 1, 2, 3, 7, 100, 2 ** 31, 2 ** 62, 2 ** 63 - 1]
ARGS = [0, 1, -1, 2, -7, 13, 2 ** 40, -(2 ** 62), 2 ** 63 - 1, -(2 ** 63)]


class Generator:
    def __init__(self, rng):
        self.rng = rng

    def expr(self, names, loopvars, depth=0):
        rng = self.rng
        r = rng.random()
        if depth > 2 or r < 0.3:
            if names and rng.random() < 0.7:
                return rng.choice(sorted(names))
            return str(rng.choice(CONSTANTS))
        if r < 0.38:
            return f'(-{self.expr(names, loopvars, depth + 1)})'
        op = rng.choice(['+', '-', '*', '//', '%'])
        left = self.expr(names, loopvars, depth + 1)
        if op == '*':
            # Keep CPython's big-int growth linear so slow replays stay fast.
            right = rng.choice(sorted(loopvars) + [str(c) for c in CONSTANTS])
        else:
            right = self.expr(names, loopvars, depth + 1)
        return f'({left} {op} {right})'

    def cond(self, names, loopvars, depth=0):
        rng = self.rng
        r = rng.random()
        if depth < 1 and r < 0.2:
            joiner = rng.choice([' and ', ' or '])
            return '(' + joiner.join(self.cond(names, loopvars, depth + 1) for _ in range(2)) + ')'
        if depth < 1 and r < 0.3:
            return f'not {self.cond(names, loopvars, depth + 1)}'
        if r < 0.38:
            return self.expr(names, loopvars, 2)  # integer truthiness
        ops = ['<', '<=', '>', '>=', '==', '!=']
        parts = [self.expr(names, loopvars, 2) for _ in range(rng.choice([2, 2, 3]))]
        out = parts[0]
        for p in parts[1:]:
            out += f' {rng.choice(ops)} {p}'
        return out

    def block(self, names, loopvars, indent, depth):
        rng = self.rng
        lines = []
        pad = '    ' * indent
        for _ in range(rng.randint(1, 4)):
            r = rng.random()
            if r < 0.35 or depth > 2:
                tgt = rng.choice(['a', 'b', 'c', 't'] + sorted(names))
                if tgt in names and rng.random() < 0.5:
                    op = rng.choice(['+=', '-=', '//=', '%='])
                    lines.append(f'{pad}{tgt} {op} {self.expr(names, loopvars)}')
                elif rng.random() < 0.2:
                    lines.append(f'{pad}{tgt}: int = {self.expr(names, loopvars)}')
                else:
                    lines.append(f'{pad}{tgt} = {self.expr(names, loopvars)}')
                names = names | {tgt}
            elif r < 0.6:
                v = rng.choice(['i', 'j', 'k'])
                form = rng.randint(1, 3)
                if form == 1:
                    rargs = f'{rng.randint(-3, 9)}'
                elif form == 2:
                    rargs = f'{rng.randint(-5, 5)}, {rng.randint(-3, 9)}'
                else:
                    rargs = f'{rng.randint(-9, 9)}, {rng.randint(-9, 9)}, {rng.choice([-3, -2, -1, 0, 1, 2, 3])}'
                lines.append(f'{pad}for {v} in range({rargs}):')
                body, _ = self.block(names | {v}, loopvars | {v}, indent + 1, depth + 1)
                lines += body
            elif r < 0.85:
                lines.append(f'{pad}if {self.cond(names, loopvars)}:')
                body, n1 = self.block(names, loopvars, indent + 1, depth + 1)
                lines += body
                if rng.random() < 0.3:
                    lines.append(f'{pad}elif {self.cond(names, loopvars)}:')
                    body, n3 = self.block(names, loopvars, indent + 1, depth + 1)
                    lines += body
                    n1 &= n3
                lines.append(f'{pad}else:')
                body, n2 = self.block(names, loopvars, indent + 1, depth + 1)
                lines += body
                names = names | (n1 & n2)
            else:
                lines.append(f'{pad}return {self.expr(names, loopvars)}')
        return lines, names

    def program(self, idx):
        params = ['x', 'y'][:self.rng.randint(0, 2)]
        body, names = self.block(set(params), set(), 1, 0)
        body.append(f'    return {self.expr(names, set())}')
        return f"def f{idx}({', '.join(params)}):\n" + '\n'.join(body) + '\n', len(params)


def outcome(fn, args):
    try:
        return ('ok', fn(*args))
    except Exception as exc:  # compare exception *types* (ZeroDivisionError, ValueError, ...)
        return ('raise', type(exc).__name__)


@unittest.skipUnless(shutil.which('rustc'), 'rustc required')
class DifferentialFuzz(unittest.TestCase):
    def test_random_programs_match_cpython(self):
        seed = int(os.environ.get('PYRUST_FUZZ_SEED', '20261106'))
        count = int(os.environ.get('PYRUST_FUZZ_PROGRAMS', '40'))
        rng = random.Random(seed)
        gen = Generator(rng)
        programs = [gen.program(i) for i in range(count)]
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, f'pyrust_fuzz_{seed}.py')
            with open(path, 'w') as fh:
                fh.write('\n'.join(src for src, _ in programs))
            spec = importlib.util.spec_from_file_location(f'pyrust_fuzz_{seed}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            native_programs = 0
            for i, (src, arity) in enumerate(programs):
                py = getattr(module, f'f{i}')
                fn = optimize(py, cache_dir=os.path.join(work, 'cache'))
                for _ in range(8):
                    args = [rng.choice(ARGS) for _ in range(arity)]
                    got, expected = outcome(fn, args), outcome(py, args)
                    self.assertEqual(got, expected, f'seed={seed} args={args}\n{src}\n{fn.rust_source()}')
                report = fn.report()
                self.assertEqual(report['backend'], 'rust', f'{report["reason"]}\n{src}')
                native_programs += 1
            self.assertEqual(native_programs, count)


if __name__ == '__main__':
    unittest.main()
