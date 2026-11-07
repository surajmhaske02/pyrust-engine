"""Compare a benchmark run with the latest release baseline and print a Markdown table.

Speedups (Python time / PyRust time) are compared, which is more stable on shared CI
machines than raw times. A drop of more than 30% is flagged with a GitHub warning.
Usage: python benchmarks/compare.py current.json [baseline.json]
"""
import json
from pathlib import Path
import platform
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

here = Path(__file__).resolve().parent
current = json.loads(Path(sys.argv[1]).read_text())
system = {'Linux': 'linux', 'Darwin': 'macos', 'Windows': 'windows'}.get(platform.system(), 'linux')
candidates = sorted(here.glob(f'results-v*-ci-{system}.json')) or sorted(here.glob('results-v*-ci-*.json'))
baseline_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (candidates[-1] if candidates else None)
if baseline_path is None:
    print('No baseline to compare with.')
    raise SystemExit(0)
baseline = {r['name']: r for r in json.loads(baseline_path.read_text())['results']}
print(f'### Benchmark vs {baseline_path.name} ({platform.system()}, Python {platform.python_version()})')
print()
print('| Workload | Baseline speedup | This run | Change |')
print('|---|---:|---:|---:|')
for row in current['results']:
    name = row['name']
    now = row.get('python_over_pyrust')
    before = baseline.get(name, {}).get('python_over_pyrust')
    if now is None:
        continue
    if before is None:
        print(f'| {name} | new | {now:.2f}x | |')
        continue
    change = now / before - 1
    flag = ' (slower)' if change < -0.30 and before > 1.5 else ''
    print(f'| {name} | {before:.2f}x | {now:.2f}x | {change:+.0%}{flag} |')
    if flag:
        print(f'::warning::{name} speedup dropped from {before:.2f}x to {now:.2f}x', file=sys.stderr)
