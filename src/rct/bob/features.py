from __future__ import annotations
from dataclasses import dataclass
import torch
from .gate import EligibilityPreview
from .types import ActionCandidate,ActionKind
FEATURE_NAMES=(
'bias','pressure_ratio','contribution_ratio','usage_ema','probe_gain_ratio','evidence_freshness','rank_fraction','probe_budget_fraction','edit_budget_fraction',
'probe_charge_fraction','edit_charge_fraction','maintenance_delta_fraction','reopen_liability_fraction','total_immediate_charge_fraction',
'route_profile_defined','delta_retained','delta_reachable','post_probe_headroom','post_formable_nonretire_class_fraction','delta_valid_handle_fraction','reachable_route_concentration_after','produces_evidence_flag',
'edge_age_fraction','generation_fraction','dormancy_version_fraction','historical_support_ratio','time_since_related_action_fraction','accepted_related_fraction','rejected_related_fraction','prior_correction_exercised',
'action_PROBE','action_CREATE','action_MODIFY_UP','action_MODIFY_DOWN','action_DORMANT','action_RETIRE','action_REOPEN','action_ABSTAIN')
FUTURE_INDICES=tuple(range(14,22))
def _clip(x,lo,hi): return max(lo,min(hi,float(x)))
def signed_ratio(x,scale,eps=1e-6): return _clip(float(x)/max(abs(float(scale)),eps),-2,2)
@dataclass(frozen=True)
class FeatureContext:
    step:int
    preview:EligibilityPreview
    full_loss_ema:float=0.0
    inquiry_pressure:float=0.0
    revision_pressure:float=0.0
    contribution_ema:float=0.0
    usage_ema:float=0.0
    probe_gain:float=0.0
    probe_baseline_mse:float=0.0
    evidence_expires_step:int|None=None
    active_rank:int=0
    probe_budget_remaining:int=2
    edit_budget_remaining:float=0.5
    route_count:int=0
    retained_before:float|None=None
    reachable_before:float|None=None
    retained_after:float|None=None
    reachable_after:float|None=None
    post_probe_remaining:int=2
    post_formable_nonretire_classes:int=1
    valid_handles_before:int=0
    valid_handles_after:int=0
    total_reachable_routes_after:int=0
    max_routes_one_commitment_after:int=0
    edge_age:int=0
    generation:int=0
    dormancy_version:int=0
    historical_support:float|None=None
    steps_since_related_action:int|None=None
    accepted_related:int=0
    rejected_related:int=0
    prior_correction_exercised:bool=False

def action_features(c:ActionCandidate,ctx:FeatureContext)->torch.Tensor:
    p=ctx.preview; f=[0.0]*38; f[0]=1.0
    pressure=ctx.inquiry_pressure if c.kind in (ActionKind.PROBE,ActionKind.CREATE) else (0.0 if c.kind is ActionKind.ABSTAIN else ctx.revision_pressure)
    f[1]=signed_ratio(pressure,ctx.full_loss_ema); f[2]=signed_ratio(ctx.contribution_ema,ctx.full_loss_ema) if c.generation is not None else 0.0; f[3]=_clip(ctx.usage_ema,0,1) if c.generation is not None else 0.0
    f[4]=signed_ratio(ctx.probe_gain,ctx.probe_baseline_mse) if ctx.probe_baseline_mse else 0.0; f[5]=_clip((ctx.evidence_expires_step-ctx.step)/64,0,1) if ctx.evidence_expires_step is not None else 0.0; f[6]=(ctx.active_rank/8) if ctx.active_rank else 0.0; f[7]=_clip(ctx.probe_budget_remaining/2,0,1); f[8]=_clip(ctx.edit_budget_remaining/.5,0,1)
    f[9]=p.probe_charge/.5; f[10]=p.edit_charge/.5; f[11]=_clip(p.maintenance_delta/.5,-2,2); f[12]=p.reopen_liability/.5; f[13]=p.total_immediate_charge/.5
    f[14]=1.0 if ctx.route_count>0 else 0.0
    if ctx.route_count>0:
        f[15]=float((ctx.retained_after or 0)-(ctx.retained_before or 0)); f[16]=float((ctx.reachable_after or 0)-(ctx.reachable_before or 0))
    f[17]=_clip(ctx.post_probe_remaining/2,0,1); f[18]=_clip(ctx.post_formable_nonretire_classes/7,0,1); f[19]=_clip((ctx.valid_handles_after-ctx.valid_handles_before)/12,-1,1); f[20]=0.0 if ctx.total_reachable_routes_after==0 else ctx.max_routes_one_commitment_after/ctx.total_reachable_routes_after; f[21]=1.0 if c.kind is ActionKind.PROBE else 0.0
    f[22]=_clip(ctx.edge_age/2048,0,1); f[23]=_clip(ctx.generation/8,0,1); f[24]=_clip(ctx.dormancy_version/8,0,1); f[25]=0.0 if ctx.historical_support is None else _clip(ctx.historical_support/max(abs(ctx.full_loss_ema),1e-6),0,2); f[26]=0.0 if c.kind is ActionKind.ABSTAIN else (1.0 if ctx.steps_since_related_action is None else _clip(ctx.steps_since_related_action/256,0,1)); f[27]=0.0 if c.kind is ActionKind.ABSTAIN else _clip(ctx.accepted_related/8,0,1); f[28]=0.0 if c.kind is ActionKind.ABSTAIN else _clip(ctx.rejected_related/8,0,1); f[29]=1.0 if ctx.prior_correction_exercised else 0.0
    action_order=(ActionKind.PROBE,ActionKind.CREATE,ActionKind.MODIFY_UP,ActionKind.MODIFY_DOWN,ActionKind.DORMANT,ActionKind.RETIRE,ActionKind.REOPEN,ActionKind.ABSTAIN); f[30+action_order.index(c.kind)]=1.0
    return torch.tensor(f,dtype=torch.float64)
def zero_future_block(phi):
    out=phi.clone(); out[list(FUTURE_INDICES)]=0.; return out

@dataclass(frozen=True)
class FutureDryRun:
    cooldown_after:int
    probe_remaining_after:int
    edit_budget_after:float
    valid_handles_after:int
    retained_after:float|None
    reachable_after:float|None
    formable_nonretire_class_count:int
    produces_evidence:bool

def future_dry_run(candidate:ActionCandidate, gate, ctx, *, route_count:int, retained_before:float|None, reachable_before:float|None)->FutureDryRun:
    """Deterministic metadata-only one-action dry run."""
    preview=gate.preview_any(candidate,ctx)
    if not preview.formable:
        return FutureDryRun(ctx.cooldown_remaining,ctx.probe_budget_remaining,ctx.edit_budget_remaining,sum(1 for h in ctx.handle_views if h.state.value=='VALID'),retained_before,reachable_before,1,False)
    structural=candidate.kind in {ActionKind.CREATE,ActionKind.MODIFY_UP,ActionKind.MODIFY_DOWN,ActionKind.DORMANT,ActionKind.RETIRE,ActionKind.REOPEN}
    cooldown_after=1 if structural else ctx.cooldown_remaining
    probe_after=max(0,ctx.probe_budget_remaining-(1 if candidate.kind is ActionKind.PROBE else 0))
    edit_after=max(0.0,ctx.edit_budget_remaining-preview.edit_charge-preview.reopen_liability)
    valid=sum(1 for h in ctx.handle_views if h.state.value=='VALID')
    if candidate.kind is ActionKind.DORMANT:
        valid+=1
    elif candidate.kind is ActionKind.REOPEN:
        valid=max(0,valid-1)
    elif candidate.kind is ActionKind.RETIRE and candidate.generation is not None:
        valid-=sum(1 for h in ctx.handle_views if h.state.value=='VALID' and h.identity.source==candidate.source and h.identity.target==candidate.target and h.identity.generation==candidate.generation)
        valid=max(0,valid)
    retained_after=retained_before
    reachable_after=(0.0 if structural and route_count>0 else reachable_before)
    classes=1
    if probe_after>0 and ctx.probe_rows_available>=256 and any(p.inquiry_qualifying_windows>=3 for p in ctx.pressures):
        classes+=1
    return FutureDryRun(cooldown_after,probe_after,edit_after,valid,retained_after,reachable_after,classes,candidate.kind is ActionKind.PROBE)
