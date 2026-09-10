import dataclasses, pytest
from rct.bob.config import BobConfig
from rct.bob.gate import StructuralGate,GateContext,TargetPressure,EdgeView,EvidenceView,apply_accepted_proposal
from rct.bob.interfaces import AdapterGraph
from rct.bob.lineage import StructuralLedger,EdgeIdentity
from rct.bob.probe import EvidenceRegistry
from rct.bob.types import ActionCandidate,ActionKind,ExistingEdgeEvidence,ProbeEvidence,StructuralProposal,EditSpec,Operation,EdgeStatus

def pressures(inquiry=3,revision=3): return tuple(TargetPressure(n,.2,.2,inquiry,revision) for n in ('A','B','C','D'))
def base_ctx(*,edge_views=(),evidence_views=(),cooldown=0,probe_remaining=2,edit_remaining=.5,retire_supported=frozenset(),applied=frozenset(),probe_rows=256):
    return GateContext(64,probe_remaining,edit_remaining,cooldown,pressures(),tuple(edge_views),(),tuple(evidence_views),frozenset(retire_supported),frozenset(applied),probe_rows)
def test_probe_and_abstain_survive_cooldown_only_when_probe_data_ready():
    g=StructuralGate(BobConfig()); ctx=base_ctx(cooldown=1); assert g.preview(ActionCandidate('P',ActionKind.PROBE,'A','B'),ctx).formable; assert g.preview(ActionCandidate('A',ActionKind.ABSTAIN),ctx).formable
    assert not g.preview(ActionCandidate('P2',ActionKind.PROBE,'A','C'),base_ctx(cooldown=1,probe_rows=255)).formable
def test_create_requires_fresh_matching_probe_evidence():
    g=StructuralGate(BobConfig()); ev=ProbeEvidence('E1','A','B',.2,1,.8,.02,64,128); ctx=base_ctx(evidence_views=(EvidenceView(ev,False),)); assert g.preview(ActionCandidate('C',ActionKind.CREATE,'A','B',evidence_ref='E1'),ctx).formable
def test_retire_requires_support():
    g=StructuralGate(BobConfig()); ident=EdgeIdentity('A','B',1); edge=EdgeView(ident,EdgeStatus.ACTIVE,2,8,100,.5,-.2); ev=ExistingEdgeEvidence('E2','A','B',1,.3,-.2,.5,2,64,128); c=ActionCandidate('R',ActionKind.RETIRE,'A','B',1,evidence_ref='E2'); ctx=base_ctx(edge_views=(edge,),evidence_views=(EvidenceView(ev,False),)); assert not g.preview(c,ctx).formable; assert g.preview(c,dataclasses.replace(ctx,retire_supported_edges=frozenset({ident}))).formable
def test_exact_once_transaction():
    cfg=BobConfig(hidden_dim=8,interface_max_rank=4,interface_initial_rank=2); g=StructuralGate(cfg); graph=AdapterGraph(8,4,2); reg=EvidenceRegistry(); led=StructuralLedger(); ev=ProbeEvidence('E1','A','B',.2,1,.8,.02,64,128); reg.add(ev); ctx=base_ctx(evidence_views=(EvidenceView(ev,False),)); p=StructuralProposal('P1',Operation.CREATE,EditSpec('A','B'),'E1'); d=g.decide(p,ctx); assert d.allowed; tx=apply_accepted_proposal(graph,reg,led,p,d); assert reg.consumed('E1'); before=graph.structural_fingerprint();
    with pytest.raises(ValueError): apply_accepted_proposal(graph,reg,led,p,d)
    assert graph.structural_fingerprint()==before and led.transactions==(tx,)
