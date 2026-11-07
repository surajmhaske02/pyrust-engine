import argparse
import inspect
import json
import runpy
import sys
from pathlib import Path

from .cache import prune_cache
from .profiling import profile_script


def check_file(path, max_iterations=10_000_000):
    """Load a trusted file (without running its `if __name__ == '__main__'` block) and
    explain, for every function defined in it, whether PyRust can compile it."""
    from .runtime import optimize

    path = Path(path).resolve()
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(path.parent))
        namespace = runpy.run_path(str(path), run_name='__pyrust_check__')
    finally:
        sys.path[:] = old_path
    results = []
    for value in namespace.values():
        original = getattr(value, 'python', value)
        if not inspect.isfunction(original) or Path(original.__code__.co_filename).resolve() != path:
            continue
        text = optimize(original, max_iterations=max_iterations, warn_on_fallback=False).explain()
        results.append(
            {
                'function': original.__qualname__,
                'line': original.__code__.co_firstlineno,
                'native': ' compiles to Rust.' in text.splitlines()[0],
                'explanation': text,
            }
        )
    return sorted(results, key=lambda r: r['line'])


def main():
    parser = argparse.ArgumentParser(prog='python -m pyrust_engine', description='PyRust Engine developer tools')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Check Python, Rust, the C linker and NumPy, with one real test compile')
    p = commands.add_parser('build', help='Compile every function in a trusted file ahead of time into __pyrust__/')
    p.add_argument('script')
    p.add_argument('--out', help='output directory (default: __pyrust__ next to the file)')
    p = commands.add_parser('check', help='Explain which functions in a trusted file compile to Rust, and why not')
    p.add_argument('script')
    p.add_argument('--json', action='store_true')
    p = commands.add_parser('profile', help='Execute a trusted script once; rank native candidates')
    p.add_argument('script')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    p = commands.add_parser('cache', help='List old cache entries; delete only with --delete')
    p.add_argument('--directory')
    p.add_argument('--days', type=float, default=30)
    p.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        from .prebuilt import build

        rows = build(args.script, args.out)
        raise SystemExit(0 if rows and all(r['ok'] for r in rows) else 1)
    if args.command == 'doctor':
        from .toolchain import doctor

        raise SystemExit(0 if doctor() else 1)
    if args.command == 'check':
        results = check_file(args.script)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for row in results:
                print(row['explanation'])
                print()
            native = sum(r['native'] for r in results)
            print(f'{native} of {len(results)} functions compile to Rust.')
        return
    result = (
        profile_script(args.script, args.arguments)
        if args.command == 'profile'
        else prune_cache(args.directory, older_than_days=args.days, dry_run=not args.delete)
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
