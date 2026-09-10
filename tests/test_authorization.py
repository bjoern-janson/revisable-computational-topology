import pytest
from rct.bob.authorization import RepoState,verify_execution_authorization,DesignCommitMismatch,ImplementationCommitMismatch,DirtyExecutableState,MalformedAuthorization
from rct.bob.types import ExecutionAuthorization
from rct.bob import DESIGN_COMMIT

def good(): return ExecutionAuthorization(DESIGN_COMMIT,'a'*40,'AUTH-BOB-MKII-001')
def state(**kw):
    d=dict(head_sha='a'*40,dirty_tracked=False,untracked_executable=()) ; d.update(kw); return RepoState(**d)
def test_wrong_design_commit_refuses():
    a=good(); a=ExecutionAuthorization('deadbeef',a.implementation_commit,a.authorization_id)
    with pytest.raises(DesignCommitMismatch): verify_execution_authorization(a,state())
def test_wrong_implementation_commit_refuses():
    with pytest.raises(ImplementationCommitMismatch): verify_execution_authorization(good(),state(head_sha='b'*40))
def test_dirty_or_untracked_executable_refuses():
    with pytest.raises(DirtyExecutableState): verify_execution_authorization(good(),state(dirty_tracked=True))
    with pytest.raises(DirtyExecutableState): verify_execution_authorization(good(),state(untracked_executable=('src/rct/bob/x.py',)))
def test_malformed_authorization_identity_refuses():
    a=good(); a=ExecutionAuthorization(a.design_commit,a.implementation_commit,'x')
    with pytest.raises(MalformedAuthorization): verify_execution_authorization(a,state())

def test_scientific_output_path_is_not_created_before_authorization(tmp_path):
    from rct.bob.lifetime import BobLifetime, ScientificExecutionNotAuthorized
    out = tmp_path / 'scientific-run'
    b = BobLifetime(conformance_only=False, output_dir=out)
    assert not out.exists()
    with pytest.raises(ScientificExecutionNotAuthorized):
        b.run_lifetime(None)
    assert not out.exists()

def test_lifetime_reads_checkout_state_inside_runner(monkeypatch,tmp_path):
    from rct.bob.lifetime import BobLifetime, ScientificExecutionNotAuthorized
    calls=[]
    def fake_from_git(cls,root='.'):
        calls.append(str(root))
        return RepoState('b'*40,False,())
    monkeypatch.setattr(RepoState,'from_git',classmethod(fake_from_git))
    b=BobLifetime(conformance_only=False,output_dir=tmp_path/'out')
    auth=ExecutionAuthorization(DESIGN_COMMIT,'a'*40,'AUTH-BOB-MKII-001')
    with pytest.raises(ScientificExecutionNotAuthorized,match='authorized implementation'):
        b.run_lifetime(auth,repo_root=tmp_path)
    assert calls==[str(tmp_path)]
    assert not (tmp_path/'out').exists()
