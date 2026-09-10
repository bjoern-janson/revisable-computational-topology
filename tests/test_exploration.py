from rct.bob.exploration import ExplorationSchedule,ExposureLedger,RetirementContextKey,RetirementSupportGate,SupportRow,select_exploration
from rct.bob.types import ActionCandidate,ActionKind

def test_exactly_one_explore_slot_per_complete_four_block():
    s=ExplorationSchedule(seed=43)
    for b in range(10): assert sum(s.is_explore(b*4+i) for i in range(4))==1

def test_exploration_never_selects_retire_and_ignores_scores():
    e=ExposureLedger(); cs=[ActionCandidate('r',ActionKind.RETIRE),ActionCandidate('a',ActionKind.ABSTAIN),ActionCandidate('p',ActionKind.PROBE,source='A',target='B')]
    c=select_exploration(cs,e)
    assert c.kind is not ActionKind.RETIRE

def test_retire_support_needs_six_rows_and_two_action_kinds_same_context():
    k=RetirementContextKey.from_values('B',.3,-.1,.7,2); g=RetirementSupportGate()
    rows=[SupportRow(k,ActionKind.MODIFY_UP) for _ in range(5)]
    assert not g.supported(k,rows)
    rows.append(SupportRow(k,ActionKind.DORMANT)); assert g.supported(k,rows)

def test_support_ignores_probe_create_abstain():
    k=RetirementContextKey.from_values('B',.3,.1,.7,2); g=RetirementSupportGate()
    rows=[SupportRow(k,ActionKind.PROBE),SupportRow(k,ActionKind.CREATE),SupportRow(k,ActionKind.ABSTAIN)]*3
    assert not g.supported(k,rows)
