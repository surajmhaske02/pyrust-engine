"""suggest / apply / MCP server: the tools for people and coding agents."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pyrust_engine import advisor, mcp

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = '''"""Sample module."""
import math


def total(n):
    s = 0
    for i in range(n):
        s += i * i
    return s


def tiny(a, b):
    return a + b


def noisy(n):
    for i in range(n):
        print(i)


def words(text: str) -> dict[str, int]:
    counts = {}
    for w in text.split():
        counts[w] = counts.get(w, 0) + 1
    return counts


if __name__ == '__main__':
    raise SystemExit('the main block must not run')
'''


class Advisor(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = Path(self.folder.name) / 'sample.py'
        self.path.write_text(SAMPLE)

    def tearDown(self):
        self.folder.cleanup()

    def test_suggest(self):
        report = advisor.suggest(self.path)
        actions = {r['function']: r['action'] for r in report['functions']}
        self.assertEqual(actions['total'], 'add @optimize')
        self.assertEqual(actions['words'], 'add @optimize')
        self.assertTrue(actions['tiny'].startswith('skip'))
        self.assertTrue(actions['noisy'].startswith('rewrite'))
        noisy = next(r for r in report['functions'] if r['function'] == 'noisy')
        self.assertEqual(noisy['problem']['line'], 18)
        self.assertEqual(noisy['problem']['code'], 'print(i)')
        self.assertEqual(report['summary']['add_optimize'], 2)
        self.assertIn('2 ready for @optimize', advisor.format_suggest(report))

    def test_apply_dry_run_then_write(self):
        result = advisor.apply(self.path)
        self.assertFalse(result['written'])
        self.assertEqual(result['decorated'], ['total', 'words'])
        self.assertIn('+@optimize', result['diff'])
        self.assertEqual(self.path.read_text(), SAMPLE)
        advisor.apply(self.path, write=True)
        text = self.path.read_text()
        self.assertIn('import math\nfrom pyrust_engine import optimize\n', text)
        self.assertIn('@optimize\ndef total(n):', text)
        self.assertNotIn('@optimize\ndef tiny', text)
        again = advisor.apply(self.path)
        self.assertEqual(again['decorated'], [])  # now already optimized
        self.assertEqual(again['diff'], '')

    def test_cli_json(self):
        out = subprocess.run(
            [sys.executable, '-m', 'pyrust_engine', 'suggest', str(self.path), '--json'],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout
        self.assertEqual(json.loads(out)['summary']['functions'], 4)


class MCP(unittest.TestCase):
    def talk(self, *messages):
        out = io.StringIO()
        mcp.main(io.StringIO('\n'.join(json.dumps(m) for m in messages) + '\n'), out)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_protocol(self):
        code = 'def f(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n'
        replies = self.talk(
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
            {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'check_code', 'arguments': {'code': code}}},
            {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'benchmark', 'arguments': {'code': code, 'call': 'f(500)'}}},
            {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'check_code', 'arguments': {}}},
            {'jsonrpc': '2.0', 'id': 6, 'method': 'nope'},
            {'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call', 'params': {'name': 'rust_source', 'arguments': {'code': code, 'function': 'f'}}},
        )
        self.assertEqual([r['id'] for r in replies], [1, 2, 3, 4, 5, 6, 7])  # no reply to the notification
        self.assertEqual(replies[0]['result']['serverInfo']['name'], 'pyrust-engine')
        names = {t['name'] for t in replies[1]['result']['tools']}
        self.assertEqual(names, {'check_code', 'suggest_file', 'apply_file', 'rust_source', 'benchmark', 'doctor'})
        checked = json.loads(replies[2]['result']['content'][0]['text'])
        self.assertTrue(checked['functions'][0]['compiles'])
        bench = json.loads(replies[3]['result']['content'][0]['text'])
        self.assertTrue(bench['same_result'])
        self.assertTrue(replies[4]['result']['isError'])
        self.assertEqual(replies[5]['error']['code'], -32601)
        self.assertIn('fn compute', json.loads(replies[6]['result']['content'][0]['text'])['rust'])


if __name__ == '__main__':
    unittest.main()
