from __future__ import annotations
import argparse,json,sys
from . import DESIGN_COMMIT,MK_I_DESIGN_AUTHORITY,ARC_SOURCE_RESULT
from .audit import audit_world
from .lifetime import BobLifetime
from .authorization import RepoState
from .types import ExecutionAuthorization

def _parser():
    p=argparse.ArgumentParser(prog='rct.bob'); sub=p.add_subparsers(dest='cmd')
    sub.add_parser('status')
    sub.add_parser('audit-world')
    v=sub.add_parser('validate'); v.add_argument('--steps',type=int,default=8)
    r=sub.add_parser('run'); r.add_argument('--execute-lifetime',action='store_true'); r.add_argument('--design-commit'); r.add_argument('--implementation-commit'); r.add_argument('--authorization-token')
    return p

def main(argv=None):
    args=_parser().parse_args(argv)
    if args.cmd=='status':
        print(f'Bob Mk II design       FROZEN @ {DESIGN_COMMIT}')
        print('Bob Mk II plan         FROZEN')
        print('Implementation         IMPLEMENTATION_IN_PROGRESS')
        print('Scientific execution  UNAUTHORIZED / UNEXECUTED')
        print('Scientific result     NONE'); return 0
    if args.cmd=='audit-world':
        r=audit_world(); print(json.dumps(r.__dict__,sort_keys=True)); return 0
    if args.cmd=='validate':
        if args.steps<0 or args.steps>64: print('validate steps must be 0..64',file=sys.stderr); return 2
        b=BobLifetime(conformance_only=True); b.run_conformance_steps(args.steps); print(f'SOFTWARE_CONFORMANCE_ONLY steps={args.steps} scientific_result=NONE'); return 0
    if args.cmd=='run':
        if not args.execute_lifetime or not args.design_commit or not args.implementation_commit or not args.authorization_token:
            print('scientific run requires explicit --execute-lifetime and full authorization identity',file=sys.stderr); return 2
        auth=ExecutionAuthorization(args.design_commit,args.implementation_commit,args.authorization_token)
        try:
            b=BobLifetime(conformance_only=False); b.run_lifetime(auth,RepoState.from_git('.'))
        except Exception as e:
            print(f'RUN_REFUSED:{type(e).__name__}:{e}',file=sys.stderr); return 3
        return 0
    _parser().print_help(); return 2
if __name__=='__main__': raise SystemExit(main())
