"""Generate docs/benchmark-speedup.svg from a benchmark_suite JSON file."""
import json, math, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
LABELS = {'integer_loop': 'Integer loop', 'float_recurrence': 'Float recurrence',
          'array_energy': 'Array kernel (100k)', 'tiny_scalar': 'Tiny add(a, b)',
          'json_array_pipeline': 'JSON → array pipeline'}
SERIES = [('pyrust', 'PyRust', 's1', 'circle')]
rows = []
for r in d['results']:
    py = r['python']['median_seconds']
    vals = {k: py / r[k]['median_seconds'] for k, *_ in SERIES if isinstance(r.get(k), dict)}
    rows.append((LABELS.get(r['name'], r['name']), vals))

W, left, right, top = 760, 190, 40, 70
row_h, sub = 62, 16
H = top + row_h * len(rows) + 56
lo, hi = 0.01, 200.0
def x(v): return left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - left - right)
def fmt(v): return f'{v:.3f}×' if v < 0.1 else (f'{v:.2f}×' if v < 10 else f'{v:.1f}×')
def marker(shape, cx, cy, cls):
    if shape == 'circle':  return f'<circle class="{cls} m" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>'
    if shape == 'square':  return f'<rect class="{cls} m" x="{cx-4.5:.1f}" y="{cy-4.5:.1f}" width="9" height="9" rx="1.5"/>'
    return f'<path class="{cls} m" d="M{cx:.1f},{cy-6:.1f} L{cx+6:.1f},{cy:.1f} L{cx:.1f},{cy+6:.1f} L{cx-6:.1f},{cy:.1f} Z"/>'
o = []
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">')
o.append('<title id="t">Speedup versus plain Python (log scale)</title>')
o.append('<desc id="d">' + '; '.join(f'{n}: ' + ', '.join(f'{lbl} {fmt(v[k])}' for k, lbl, *_ in SERIES if k in v) for n, v in rows) + '</desc>')
o.append('''<style>
.bg{fill:#fcfcfb}.t1{fill:#21201c;font:600 15px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
.t2{fill:#5f5e58;font:12px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
.lab{fill:#21201c;font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
.val{fill:#5f5e58;font:11px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
.grid{stroke:#e6e5e0;stroke-width:1}.ref{stroke:#8c8b84;stroke-width:1.5;stroke-dasharray:4 3}
.stem{stroke:#d4d3cc;stroke-width:2;stroke-linecap:round}
.m{stroke:#fcfcfb;stroke-width:2}.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}
@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1,.lab{fill:#f0efec}.t2,.val{fill:#c3c2b7}
.grid{stroke:#2e2d2a}.ref{stroke:#8c8b84}.stem{stroke:#3a3936}.m{stroke:#1a1a19}
.s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}}
</style>''')
o.append(f'<rect class="bg" width="{W}" height="{H}" rx="8"/>')
o.append(f'<text class="t1" x="20" y="28">PyRust speedup versus plain Python — higher is faster</text>')
# legend
lx = 20
for k, lbl, cls, shape in SERIES:
    o.append(marker(shape, lx + 6, 48, cls)); o.append(f'<text class="t2" x="{lx+16}" y="52">{lbl}</text>'); lx += 26 + len(lbl) * 7.2
ybot = top + row_h * len(rows)
for t in [0.01, 0.1, 1, 10, 100]:
    cls = 'ref' if t == 1 else 'grid'
    o.append(f'<line class="{cls}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{ybot}"/>')
    lbl = {0.01: '0.01×', 0.1: '0.1×', 1: '1× = Python', 10: '10×', 100: '100×'}[t]
    o.append(f'<text class="t2" x="{x(t):.1f}" y="{ybot+18}" text-anchor="middle">{lbl}</text>')
o.append(f'<text class="t2" x="{W-right}" y="{ybot+38}" text-anchor="end">Log scale · median of 7 warm runs · Linux x86_64, Python 3.13, Rust 1.97 · PyRust 0.3.0</text>')
for i, (name, vals) in enumerate(rows):
    y0 = top + i * row_h
    o.append(f'<text class="lab" x="20" y="{y0 + row_h/2 + 4:.1f}">{name}</text>')
    present = [s for s in SERIES if s[0] in vals]
    offset = (row_h - sub * len(present)) / 2 + sub / 2
    for j, (k, lbl, cls, shape) in enumerate(present):
        cy = y0 + offset + j * sub; v = vals[k]; cx = x(v)
        o.append(f'<line class="stem" x1="{x(1):.1f}" x2="{cx:.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>')
        o.append(marker(shape, cx, cy, cls))
        anchor, dx = ('end', -10) if v < 1 else ('start', 10)
        o.append(f'<text class="val" x="{cx+dx:.1f}" y="{cy+4:.1f}" text-anchor="{anchor}">{fmt(v)}</text>')
o.append('</svg>')
open(dst, 'w').write('\n'.join(o))
