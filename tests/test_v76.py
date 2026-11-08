"""0.7.6: suggest/apply and the MCP server see class methods; MCP 2025-11-25."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from pyrust_engine import advisor, mcp

CLASSES = '''"""Classes for the advisor."""
from dataclasses import dataclass


@dataclass
class Particle:
    x: float
    vx: float
    steps: int = 0

    def advance(self, dt: float, n: int) -> None:
        for _ in range(n):
            self.x += self.vx * dt
            self.steps += 1

    def calls(self) -> float:
        total = 0.0
        for _ in range(3):
            total += self.speed()
        return total

    def speed(self) -> float:
        return abs(self.vx)

    @staticmethod
    def helper(n):
        t = 0
        for i in range(n):
            t += i
        return t


class Account:
    def __init__(self, balance: float):
        self.balance = balance
        self.rate = 0.01

    def grow(self, years: int) -> float:
        for _ in range(years):
            self.balance += self.balance * self.rate
        return self.balance

    def mystery(self, n: int) -> int:
        t = 0
        for i in range(n):
            t += self.unknown
        return t


def plain(n):
    t = 0
    for i in range(n):
        t += i
    return t
'''


class Methods(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self.folder.name) / 'shapes.py'
        self.path.write_text(CLASSES)

    def tearDown(self):
        self.folder.cleanup()

    def test_suggest(self):
        rows = {r['function']: r for r in advisor.suggest(self.path)['functions']}
        self.assertEqual(rows['Particle.advance']['action'], 'add @optimize')
        self.assertEqual(rows['Account.grow']['action'], 'add @optimize')  # types from __init__
        self.assertEqual(rows['plain']['action'], 'add @optimize')
        self.assertTrue(rows['Particle.speed']['action'].startswith('skip'))
        self.assertIn('plain function', rows['Particle.calls']['problem']['hint'])
        self.assertIn('self.unknown', rows['Account.mystery']['problem']['message'])
        self.assertNotIn('Particle.helper', rows)  # staticmethods are left alone

    def test_apply(self):
        result = advisor.apply(self.path, write=True)
        self.assertEqual(result['decorated'], ['Account.grow', 'Particle.advance', 'plain'])
        text = self.path.read_text()
        self.assertIn('    @optimize\n    def advance(self, dt: float, n: int) -> None:', text)
        self.assertIn('    @optimize\n    def grow(self, years: int) -> float:', text)
        self.assertIn('@optimize\ndef plain(n):', text)
        self.assertIn('from dataclasses import dataclass\nfrom pyrust_engine import optimize\n', text)
        compile(text, str(self.path), 'exec')
        self.assertEqual(advisor.apply(self.path)['decorated'], [])  # now already decorated

    def test_mcp_check_code(self):
        rows = {r['function']: r for r in mcp.check_code(CLASSES)['functions']}
        self.assertTrue(rows['Particle.advance']['compiles'])
        self.assertTrue(rows['Account.grow']['compiles'])
        self.assertFalse(rows['Particle.calls']['compiles'])
        self.assertIn('fn compute', mcp.rust_source(CLASSES, 'Particle.advance')['rust'])


class Protocol(unittest.TestCase):
    def talk(self, *messages):
        out = io.StringIO()
        mcp.main(io.StringIO('\n'.join(json.dumps(m) for m in messages) + '\n'), out)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_versions_hints_and_structured_results(self):
        replies = self.talk(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-11-25'}},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
            {'jsonrpc': '2.0', 'id': 3, 'method': 'initialize', 'params': {'protocolVersion': '1999-01-01'}},
            {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/list'},
            {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'doctor', 'arguments': {}}},
        )
        self.assertEqual([r['result']['protocolVersion'] for r in replies[:3]], ['2025-11-25', '2024-11-05', '2025-11-25'])
        tools = {t['name']: t for t in replies[3]['result']['tools']}
        self.assertTrue(tools['apply_file']['annotations']['destructiveHint'])
        self.assertTrue(tools['doctor']['annotations']['readOnlyHint'])
        self.assertFalse(tools['check_code']['annotations']['readOnlyHint'])  # it runs the given code
        self.assertTrue(all(t['title'] for t in tools.values()))
        result = replies[4]['result']
        self.assertEqual(result['structuredContent'], json.loads(result['content'][0]['text']))


if __name__ == '__main__':
    unittest.main()
