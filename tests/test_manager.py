from rct.bob.manager import CandidateGenerator,TopologyManager
from rct.bob.config import BobConfig,MkIIConfig
from rct.bob.gate import StructuralGate,GateContext,TargetPressure
from rct.bob.types import ActionKind
from rct.bob.features import FeatureContext

def _ctx(rows=256):
    ps=tuple(TargetPressure(n,.2,.2,3,3) for n in ('A','B','C','D'))
    return GateContext(t=512,probe_budget_remaining=2,edit_budget_remaining=.5,cooldown_remaining=0,pressures=ps,edge_views=(),handle_views=(),evidence_views=(),retire_supported_edges=frozenset(),applied_proposal_ids=frozenset(),probe_rows_available=rows)

def test_generator_emits_at_most_two_qualified_absent_pairs_without_self_edges():
    g=CandidateGenerator(seed=31,max_candidates=2); xs=g.generate(_ctx())
    assert len(xs)==2 and all(c.kind is ActionKind.PROBE and c.source!=c.target for c in xs)

def test_probe_not_candidate_when_buffer_not_ready():
    m=TopologyManager(StructuralGate(BobConfig()),MkIIConfig()); cs=m.build_candidates(_ctx(rows=255))
    assert [c.kind for c in cs]==[ActionKind.ABSTAIN]

def test_manager_choice_has_no_gate_permission_and_keeps_complete_features():
    m=TopologyManager(StructuralGate(BobConfig()),MkIIConfig()); cs=m.build_candidates(_ctx()); f={c.candidate_id:FeatureContext(step=512,preview=m.gate.preview_any(c,_ctx())) for c in cs}; ch=m.choose_action(cs,f,decision_index=1)
    assert ch.gate_decision is None
    assert set(ch.features)=={c.candidate_id for c in cs}

def test_explore_never_forces_retire():
    m=TopologyManager(StructuralGate(BobConfig()),MkIIConfig()); cs=m.build_candidates(_ctx()); f={c.candidate_id:FeatureContext(step=512,preview=m.gate.preview_any(c,_ctx())) for c in cs}
    idx=next(i for i in range(4) if m.schedule.is_explore(i)); ch=m.choose_action(cs,f,decision_index=idx); assert ch.candidate.kind is not ActionKind.RETIRE
