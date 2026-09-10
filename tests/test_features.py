import torch
from rct.bob.features import FEATURE_NAMES,FUTURE_INDICES,FeatureContext,action_features,zero_future_block
from rct.bob.types import ActionCandidate,ActionKind
from rct.bob.gate import EligibilityPreview

def test_exact_feature_schema_and_action_one_hot():
    assert len(FEATURE_NAMES)==38
    assert FEATURE_NAMES[14:22]==('route_profile_defined','delta_retained','delta_reachable','post_probe_headroom','post_formable_nonretire_class_fraction','delta_valid_handle_fraction','reachable_route_concentration_after','produces_evidence_flag')
    c=ActionCandidate('x',ActionKind.ABSTAIN)
    phi=action_features(c,FeatureContext(step=256,preview=EligibilityPreview(c,True,'ok')))
    assert phi.shape==(38,) and phi.dtype==torch.float64
    assert phi[37]==1 and phi[30:37].sum()==0

def test_zero_future_block_only_zeros_14_through_21():
    x=torch.arange(38,dtype=torch.float64); y=zero_future_block(x)
    assert torch.equal(y[:14],x[:14]) and torch.equal(y[22:],x[22:]) and torch.equal(y[14:22],torch.zeros(8,dtype=torch.float64))

def test_metadata_dry_run_can_encode_cooldown_reachability_drop_without_mutation():
    c=ActionCandidate('d',ActionKind.DORMANT,source='A',target='B',generation=1)
    ctx=FeatureContext(step=512,preview=EligibilityPreview(c,True,'ok',edit_charge=.05,maintenance_delta=-.002),route_count=2,retained_before=1.0,reachable_before=1.0,retained_after=1.0,reachable_after=0.0,post_probe_remaining=2,post_formable_nonretire_classes=1,valid_handles_before=0,valid_handles_after=1,total_reachable_routes_after=0,max_routes_one_commitment_after=0,full_loss_ema=.5,revision_pressure=.2,active_rank=2,generation=1)
    phi=action_features(c,ctx)
    assert phi[15]==0.0 and phi[16]==-1.0 and phi[19]>0

def test_future_dry_run_uses_metadata_only_and_accounts_for_structural_cooldown():
    from rct.bob.features import future_dry_run
    from rct.bob.gate import GateContext,TargetPressure,EdgeView,EvidenceView
    from rct.bob.lineage import EdgeIdentity
    from rct.bob.types import ExistingEdgeEvidence,EdgeStatus
    from rct.bob.config import BobConfig
    from rct.bob.gate import StructuralGate
    ps=tuple(TargetPressure(n,.2,.2,3,3) for n in ('A','B','C','D'))
    e=EdgeView(EdgeIdentity('A','B',1),EdgeStatus.ACTIVE,2,8)
    ev=ExistingEdgeEvidence('E','A','B',1,.2,-.1,.4,2,512,576)
    ctx=GateContext(512,2,.5,0,ps,(e,),(),(EvidenceView(ev,False),),frozenset(),frozenset(),256)
    c=ActionCandidate('D',ActionKind.DORMANT,'A','B',1,evidence_ref='E')
    before=repr(ctx)
    d=future_dry_run(c,StructuralGate(BobConfig()),ctx,route_count=2,retained_before=1.0,reachable_before=1.0)
    assert d.cooldown_after==1 and d.reachable_after==0.0 and d.valid_handles_after==1
    assert repr(ctx)==before

def test_future_dry_run_counts_all_currently_formable_nonretire_classes():
    from rct.bob.features import future_dry_run
    from rct.bob.gate import GateContext,TargetPressure,EdgeView,EvidenceView
    from rct.bob.lineage import EdgeIdentity
    from rct.bob.types import ExistingEdgeEvidence,EdgeStatus,ActionCandidate,ActionKind
    from rct.bob.config import BobConfig
    from rct.bob.gate import StructuralGate
    ps=tuple(TargetPressure(n,.2,.2,3,3) for n in ('A','B','C','D'))
    e=EdgeView(EdgeIdentity('A','B',1),EdgeStatus.ACTIVE,2,8)
    ev=ExistingEdgeEvidence('E','A','B',1,.2,.1,.4,2,512,576)
    ctx=GateContext(512,2,.5,0,ps,(e,),(),(EvidenceView(ev,False),),frozenset(),frozenset(),256)
    c=ActionCandidate('ABSTAIN',ActionKind.ABSTAIN)
    d=future_dry_run(c,StructuralGate(BobConfig()),ctx,route_count=0,retained_before=None,reachable_before=None)
    assert d.formable_nonretire_class_count==5
