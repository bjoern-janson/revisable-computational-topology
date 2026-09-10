from pathlib import Path
from rct.bob.cli import main

def test_status_reports_unexecuted(capsys):
    assert main(['status'])==0; s=capsys.readouterr().out
    assert 'AUTHORIZED / UNEXECUTED' in s and 'Scientific result     NONE' in s

def test_validate_is_non_scientific_and_creates_no_result_dirs(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path); assert main(['validate','--steps','2'])==0
    assert not (tmp_path/'runs').exists() and not (tmp_path/'results').exists()

def test_run_requires_explicit_execute_flag():
    assert main(['run'])!=0

def test_scientific_run_requires_explicit_output_directory(capsys):
    from rct.bob import DESIGN_COMMIT
    rc=main(['run','--execute-lifetime','--design-commit',DESIGN_COMMIT,
             '--implementation-commit','a'*40,'--authorization-token','AUTH-BOB-MKII-001'])
    assert rc==2
    assert 'output' in capsys.readouterr().err.lower()

def test_cli_does_not_inject_repo_state_into_scientific_runner(monkeypatch,tmp_path):
    from rct.bob import DESIGN_COMMIT
    calls=[]
    def fake_run(self,authorization,repo_root='.'):
        calls.append((authorization,repo_root))
        return []
    monkeypatch.setattr('rct.bob.lifetime.BobLifetime.run_lifetime',fake_run)
    rc=main(['run','--execute-lifetime','--design-commit',DESIGN_COMMIT,
             '--implementation-commit','a'*40,'--authorization-token','AUTH-BOB-MKII-001',
             '--output-dir',str(tmp_path/'out')])
    assert rc==0
    assert len(calls)==1
    auth,root=calls[0]
    assert auth.implementation_commit=='a'*40
    assert root=='.'
