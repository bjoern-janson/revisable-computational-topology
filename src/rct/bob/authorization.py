from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re,subprocess
from . import DESIGN_COMMIT
from .types import ExecutionAuthorization
class AuthorizationError(RuntimeError): pass
class DesignCommitMismatch(AuthorizationError): pass
class ImplementationCommitMismatch(AuthorizationError): pass
class DirtyExecutableState(AuthorizationError): pass
class MalformedAuthorization(AuthorizationError): pass
@dataclass(frozen=True)
class RepoState:
    head_sha:str
    dirty_tracked:bool
    untracked_executable:tuple[str,...]
    @classmethod
    def from_git(cls,root='.'):
        root=Path(root); head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(); out=subprocess.check_output(['git','status','--porcelain','-uall'],cwd=root,text=True); dirty=False; un=[]
        for line in out.splitlines():
            code=line[:2]; path=line[3:]
            if code=='??':
                if path.startswith('src/') and path.endswith('.py'): un.append(path)
            elif path.startswith('src/') or path.startswith('pyproject.toml'): dirty=True
        return cls(head,dirty,tuple(sorted(un)))
def verify_execution_authorization(auth:ExecutionAuthorization|None,repo_state:RepoState):
    if auth is None: raise MalformedAuthorization('authorization required')
    if auth.design_commit!=DESIGN_COMMIT: raise DesignCommitMismatch(f'{auth.design_commit} != {DESIGN_COMMIT}')
    if not re.fullmatch(r'[0-9a-f]{40}',auth.implementation_commit): raise ImplementationCommitMismatch('implementation commit must be full lowercase SHA')
    if auth.implementation_commit!=repo_state.head_sha: raise ImplementationCommitMismatch('checked-out HEAD does not match authorized implementation')
    if not isinstance(auth.authorization_id,str) or len(auth.authorization_id)<8 or not auth.authorization_id.startswith('AUTH-'): raise MalformedAuthorization('authorization identity malformed')
    if repo_state.dirty_tracked or repo_state.untracked_executable: raise DirtyExecutableState('executable source is not clean')
    return True
