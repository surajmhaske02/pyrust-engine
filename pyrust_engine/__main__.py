import argparse
import json
from .profiling import profile_script
from .cache import prune_cache

def main():
    parser=argparse.ArgumentParser(description='PyRust Engine diagnostics')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('profile',help='Execute a trusted script once; rank native candidates')
    p.add_argument('script');p.add_argument('arguments',nargs=argparse.REMAINDER)
    p=commands.add_parser('cache',help='List old cache entries; delete only with --delete')
    p.add_argument('--directory');p.add_argument('--days',type=float,default=30)
    p.add_argument('--delete',action='store_true')
    args=parser.parse_args()
    result=(profile_script(args.script,args.arguments) if args.command=='profile' else
            prune_cache(args.directory,older_than_days=args.days,dry_run=not args.delete))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
