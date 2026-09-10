from rct.bob.metrics import CorrectionRoute, CorrectiveAccessLedger

def test_empty_route_cohort_is_na_not_one():
    p=CorrectiveAccessLedger().profile()
    assert p.route_count==0
    assert p.retained is None and p.reachable is None

def test_persistent_route_denominator_survives_unreachable():
    l=CorrectiveAccessLedger(); l.admit(CorrectionRoute('c1','r1','REOPEN',1,'H1'),retained=True,reachable=True)
    assert l.profile().reachable==1.0
    l.set_state('r1',retained=True,reachable=False)
    p=l.profile(); assert p.route_count==1 and p.retained==1.0 and p.reachable==0.0

def test_route_ledger_exposes_stable_route_identities():
    l=CorrectiveAccessLedger(); r=CorrectionRoute('c1','r1','DORMANT',1,'DORMANT')
    l.admit(r,retained=True,reachable=False)
    assert l.routes==(r,)
