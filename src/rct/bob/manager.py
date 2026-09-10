from __future__ import annotations
from dataclasses import dataclass
import hashlib, torch
from .types import NODE_IDS,ActionCandidate,ActionKind,EdgeStatus,ProbeEvidence,ExistingEdgeEvidence,ReopenEvidence,HandleState
from .gate import StructuralGate,GateContext
from .features import FeatureContext,action_features
from .valuation import LinearActionValuation
from .exploration import ExplorationSchedule,ExposureLedger,select_exploration

@dataclass(frozen=True)
class ModuleStats:
    target:str; local_loss_ema:float; full_loss_ema:float; inquiry_pressure:float; revision_pressure:float
@dataclass(frozen=True)
class EdgeStats:
    candidate_id:str; contribution_ema:float; usage_ema:float
@dataclass(frozen=True)
class ManagerObservation:
    step:int; modules:tuple[ModuleStats,...]=(); edges:tuple[EdgeStats,...]=()
@dataclass(frozen=True)
class ChosenAction:
    candidate:ActionCandidate
    mode:str
    score:float|None
    features:dict[str,torch.Tensor]
    scores:dict[str,float]
    gate_decision:object|None=None

class CandidateGenerator:
    def __init__(self,seed:int=31,max_candidates:int=2): self._gen=torch.Generator().manual_seed(seed); self.max_candidates=max_candidates
    def generate(self,ctx:GateContext):
        qualified={p.target for p in ctx.pressures if p.inquiry_qualifying_windows>=3}
        active={(e.identity.source,e.identity.target) for e in ctx.edge_views if e.status is EdgeStatus.ACTIVE}
        pool=[(a,b) for a in NODE_IDS for b in NODE_IDS if a!=b and b in qualified and (a,b) not in active]
        if not pool:return []
        order=torch.randperm(len(pool),generator=self._gen).tolist()[:self.max_candidates]
        return [ActionCandidate(f'PROBE:{pool[i][0]}->{pool[i][1]}',ActionKind.PROBE,source=pool[i][0],target=pool[i][1]) for i in order]

class TopologyManager:
    def __init__(self,gate:StructuralGate,mkii_config,valuation:LinearActionValuation|None=None):
        self.gate=gate; self.cfg=mkii_config; self.valuation=valuation or LinearActionValuation(mkii_config.feature_dim,mkii_config.eta_ridge_lambda); self.generator=CandidateGenerator(seed=31,max_candidates=mkii_config.relation_candidates_per_boundary); self.schedule=ExplorationSchedule(mkii_config.exploration_seed); self.exposure=ExposureLedger()
    def _structural_candidates(self,ctx):
        out=[]
        for evv in ctx.evidence_views:
            if evv.consumed: continue
            ev=evv.evidence
            if isinstance(ev,ProbeEvidence): out.append(ActionCandidate(f'CREATE:{ev.source}->{ev.target}:{ev.evidence_id}',ActionKind.CREATE,ev.source,ev.target,evidence_ref=ev.evidence_id))
        for e in ctx.edge_views:
            if e.status is not EdgeStatus.ACTIVE: continue
            matches=[v.evidence for v in ctx.evidence_views if not v.consumed and isinstance(v.evidence,ExistingEdgeEvidence) and (v.evidence.source,v.evidence.target,v.evidence.generation)==(e.identity.source,e.identity.target,e.identity.generation)]
            if not matches: continue
            ev=matches[-1]; base=f'{e.identity.source}->{e.identity.target}:g{e.identity.generation}:{ev.evidence_id}'
            out.extend([
                ActionCandidate('MODIFY_UP:'+base,ActionKind.MODIFY_UP,e.identity.source,e.identity.target,e.identity.generation,evidence_ref=ev.evidence_id,rank_delta=1),
                ActionCandidate('MODIFY_DOWN:'+base,ActionKind.MODIFY_DOWN,e.identity.source,e.identity.target,e.identity.generation,evidence_ref=ev.evidence_id,rank_delta=-1),
                ActionCandidate('DORMANT:'+base,ActionKind.DORMANT,e.identity.source,e.identity.target,e.identity.generation,evidence_ref=ev.evidence_id),
                ActionCandidate('RETIRE:'+base,ActionKind.RETIRE,e.identity.source,e.identity.target,e.identity.generation,evidence_ref=ev.evidence_id),
            ])
        for h in ctx.handle_views:
            if h.state is not HandleState.VALID or not h.historical_support: continue
            matches=[v.evidence for v in ctx.evidence_views if not v.consumed and isinstance(v.evidence,ReopenEvidence) and v.evidence.handle_id==h.handle_id]
            if matches:
                ev=matches[-1]; out.append(ActionCandidate(f'REOPEN:{h.handle_id}:{ev.evidence_id}',ActionKind.REOPEN,h.identity.source,h.identity.target,h.identity.generation,h.dormancy_version,h.handle_id,ev.evidence_id))
        return out
    def build_candidates(self,ctx:GateContext):
        raw=list(self.generator.generate(ctx))+self._structural_candidates(ctx)+[ActionCandidate('ABSTAIN',ActionKind.ABSTAIN)]
        eligible=[c for c in raw if self.gate.preview_any(c,ctx).formable]
        if not any(c.kind is ActionKind.ABSTAIN for c in eligible): eligible.append(ActionCandidate('ABSTAIN',ActionKind.ABSTAIN))
        self.exposure.record_eligibility(eligible)
        return eligible
    def choose_action(self,candidates,feature_contexts:dict[str,FeatureContext],*,decision_index:int):
        phis={c.candidate_id:action_features(c,feature_contexts[c.candidate_id]) for c in candidates}
        scores={cid:self.valuation.score(phi) for cid,phi in phis.items()}
        if self.schedule.is_explore(decision_index):
            c=select_exploration(candidates,self.exposure); return ChosenAction(c,'EXPLORE',None,phis,scores,None)
        c=max(candidates,key=lambda x:(scores[x.candidate_id],-int(hashlib.sha256(f'47|{x.candidate_id}'.encode()).hexdigest(),16)))
        self.exposure.record_choice(c); return ChosenAction(c,'GREEDY',scores[c.candidate_id],phis,scores,None)
