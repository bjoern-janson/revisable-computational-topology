from __future__ import annotations
from dataclasses import replace
import hashlib
import torch
import torch.nn.functional as F
from .config import BobConfig,WorldConfig,MkIIConfig
from .types import NODE_IDS,ActionCandidate,ActionKind,EdgeStatus,CandidateRelation,ExistingEdgeEvidence,ReopenEvidence,StructuralProposal,EditSpec,Operation
from .world import RotatingDependencyWorld
from .backbone import BobBackbone
from .interfaces import AdapterGraph
from .lineage import StructuralLedger,EdgeIdentity
from .probe import ProbeBudget,LatentResidualBuffer,EvidenceRegistry,relation_probe
from .gate import StructuralGate,GateContext,TargetPressure,EdgeView,HandleView,EvidenceView
from .manager import TopologyManager
from .features import FeatureContext,signed_ratio
from .valuation import ValuationLedger,LinearActionValuation
from .metrics import CorrectiveAccessLedger,CorrectionRoute
from .exploration import RetirementSupportGate,RetirementContextKey,SupportRow

class ScientificExecutionNotAuthorized(RuntimeError): pass

class BobLifetime:
    def __init__(self,*,conformance_only:bool=False,bob_config=None,world_config=None,mkii_config=None,output_dir=None):
        from .trajectory import TrajectoryRecorder
        self.conformance_only=conformance_only; self.bob_config=bob_config or BobConfig(); self.world_config=world_config or WorldConfig(); self.mkii_config=mkii_config or MkIIConfig(); self.step_count=0
        self.world=None
        if conformance_only:
            wc=replace(self.world_config,w1_steps_min=100000,w1_steps_max=100000,w2_steps_min=1,w2_steps_max=1,w3_steps_min=1,w3_steps_max=1)
            self.world=RotatingDependencyWorld(wc)
        self.backbone=BobBackbone(self.world_config,self.bob_config); self.graph=AdapterGraph(self.bob_config.hidden_dim,self.bob_config.interface_max_rank,self.bob_config.interface_initial_rank,edge_lr=1e-3); self.backbone_opt=torch.optim.AdamW(self.backbone.parameters(),lr=3e-4)
        self.registry=EvidenceRegistry(); self.structural_ledger=StructuralLedger(); self.probe_budget=ProbeBudget(self.bob_config.probes_per_window,self.bob_config.probe_cost); self.probe_buffer=LatentResidualBuffer(self.bob_config.probe_window); self.access=CorrectiveAccessLedger(); self.valuation_ledger=ValuationLedger(self.mkii_config.return_horizon_steps); self.valuation=LinearActionValuation(self.mkii_config.feature_dim,self.mkii_config.eta_ridge_lambda); self.gate=StructuralGate(self.bob_config); self.manager=TopologyManager(self.gate,self.mkii_config,self.valuation); self._retire_support_gate=RetirementSupportGate(self.mkii_config.retire_support_min_rows,self.mkii_config.retire_support_min_action_kinds); self._retire_support_rows=[]; self._retire_support_pending={}; self._related_history={}
        self.recorder=TrajectoryRecorder(output_dir)
        self.module_ema={n:{'local':0.0,'full':0.0} for n in NODE_IDS}; self._window_local={n:[] for n in NODE_IDS}; self._prev_window_local={n:None for n in NODE_IDS}; self._inq_q={n:0 for n in NODE_IDS}; self._rev_q={n:0 for n in NODE_IDS}; self._pressures={n:(0.0,0.0) for n in NODE_IDS}; self._decision_index=0; self._cooldown=0; self.cost_history=[]; self.last_preupdate_encoder_fingerprint=''; self._last_action_meta={}
    def encoder_fingerprint(self):
        h=hashlib.sha256()
        for n in NODE_IDS:
            for p in self.backbone.encoders[n].parameters(): h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def _ema(self,old,x): return .95*old+.05*float(x)
    def _update_pressures(self,local_terms,full_terms,boundary):
        for n in NODE_IDS:
            self.module_ema[n]['local']=self._ema(self.module_ema[n]['local'],local_terms[n]); self.module_ema[n]['full']=self._ema(self.module_ema[n]['full'],full_terms[n]); self._window_local[n].append(float(local_terms[n]))
        if not boundary:return
        for n in NODE_IDS:
            cur=sum(self._window_local[n])/max(1,len(self._window_local[n])); prev=self._prev_window_local[n]; slope=999. if prev is None else cur-prev; self._prev_window_local[n]=cur; self._window_local[n]=[]
            disc=self.module_ema[n]['local'] if self.step_count>=self.bob_config.manager_warmup_steps and abs(slope)<.002 else 0.0
            incoming=[e for e in self.graph.active_edges() if e.identity.target==n]; rev=max(0.0,self.module_ema[n]['full']-self.module_ema[n]['local'])+max(0.0,-sum(e.contribution_ema for e in incoming)); inq=max(disc,rev); self._inq_q[n]=self._inq_q[n]+1 if inq>.05 else 0; self._rev_q[n]=self._rev_q[n]+1 if rev>.05 else 0; self._pressures[n]=(inq,rev)
    def _emit_edge_evidence(self,t):
        if t<self.bob_config.manager_warmup_steps:return
        for e in self.graph.active_edges():
            eid=self.registry.next_id(); ev=ExistingEdgeEvidence(eid,e.identity.source,e.identity.target,e.identity.generation,self._pressures[e.identity.target][1],e.contribution_ema,e.usage_ema,e.active_rank,t,t+64); self.registry.add(ev)
        for h in self.graph.handles():
            if h.state.value=='VALID' and h.historical_support is not None and h.historical_support>0:
                eid=self.registry.next_id(); ev=ReopenEvidence(eid,h.identity.source,h.identity.target,h.identity.generation,h.dormancy_version,h.handle_id,self._pressures[h.identity.target][1],h.historical_support,h.reopening_liability,t,t+64); self.registry.add(ev)
    def _retirement_key(self,edge):
        target=edge.identity.target
        full=self.module_ema[target]['full']
        return RetirementContextKey.from_values(
            target,
            signed_ratio(self._pressures[target][1],full),
            signed_ratio(edge.contribution_ema,full),
            edge.usage_ema,
            edge.active_rank,
        )

    def _ctx(self,t):
        ps=tuple(TargetPressure(n,*self._pressures[n],self._inq_q[n],self._rev_q[n]) for n in NODE_IDS)
        edges=tuple(EdgeView(e.identity,e.status,e.active_rank,self.graph.max_rank,e.age,e.usage_ema,e.contribution_ema) for e in self.graph.edges())
        hs=tuple(HandleView(h.handle_id,h.identity,h.dormancy_version,h.state,h.historical_support,h.reopening_liability,h.snapshot.active_rank) for h in self.graph.handles())
        evs=tuple(EvidenceView(ev,self.registry.consumed(ev.evidence_id)) for ev in self.registry.items())
        retire_supported=frozenset(
            e.identity for e in self.graph.active_edges()
            if self._retire_support_gate.supported(self._retirement_key(e),self._retire_support_rows)
        )
        return GateContext(t,self.probe_budget.remaining,self.bob_config.edit_budget_per_window,self._cooldown,ps,edges,hs,evs,retire_supported,frozenset(tx.proposal_id for tx in self.structural_ledger.transactions),probe_rows_available=len(self.probe_buffer))

    @staticmethod
    def _commitment_id(identity):
        return f'C:{identity.source}->{identity.target}:g{identity.generation}'

    @staticmethod
    def _parse_commitment_id(commitment_id):
        body=commitment_id[2:]
        pair,generation=body.rsplit(':g',1)
        source,target=pair.split('->',1)
        return source,target,int(generation)

    def _admit_route(self,commitment_id,operation,generation,target_rank_or_handle):
        route_id=f'{commitment_id}|{operation}:{target_rank_or_handle}'
        self.access.admit(CorrectionRoute(commitment_id,route_id,operation,generation,str(target_rank_or_handle)),retained=True,reachable=False)

    def _sync_corrective_access(self,t):
        for edge in self.graph.active_edges():
            if edge.usage_ema >= .25 and edge.contribution_ema >= .01:
                cid=self._commitment_id(edge.identity)
                if edge.active_rank < self.graph.max_rank:
                    self._admit_route(cid,'MODIFY_UP',edge.identity.generation,f'r{edge.active_rank+1}')
                if edge.active_rank > 1:
                    self._admit_route(cid,'MODIFY_DOWN',edge.identity.generation,f'r{edge.active_rank-1}')
                self._admit_route(cid,'DORMANT',edge.identity.generation,'DORMANT')
                self._admit_route(cid,'RETIRE',edge.identity.generation,'RETIRE')
        for handle in self.graph.handles():
            cid=self._commitment_id(handle.identity)
            if cid in self.access.commitment_ids:
                self._admit_route(cid,'REOPEN',handle.identity.generation,handle.handle_id)

        ctx=self._ctx(t)
        edges={e.identity:e for e in ctx.edge_views}
        handles={h.handle_id:h for h in ctx.handle_views}
        for route in self.access.routes:
            source,target,generation=self._parse_commitment_id(route.commitment_id)
            identity=EdgeIdentity(source,target,generation)
            edge=edges.get(identity)
            retained=False; reachable=False
            if route.operation=='REOPEN':
                handle=handles.get(route.target_rank_or_handle)
                retained=bool(handle is not None and handle.state.value=='VALID')
                if retained:
                    evs=[v.evidence for v in ctx.evidence_views if not v.consumed and isinstance(v.evidence,ReopenEvidence) and v.evidence.handle_id==handle.handle_id]
                    if evs:
                        ev=evs[-1]
                        c=ActionCandidate(route.route_id,ActionKind.REOPEN,source,target,generation,handle.dormancy_version,handle.handle_id,ev.evidence_id)
                        reachable=self.gate.preview_any(c,ctx).formable
            else:
                retained=bool(edge is not None and edge.status is not EdgeStatus.RETIRED)
                if retained and edge.status is EdgeStatus.ACTIVE:
                    evs=[v.evidence for v in ctx.evidence_views if not v.consumed and isinstance(v.evidence,ExistingEdgeEvidence) and (v.evidence.source,v.evidence.target,v.evidence.generation)==(source,target,generation)]
                    if evs:
                        ev=evs[-1]
                        if route.operation=='MODIFY_UP': kind=ActionKind.MODIFY_UP; delta=1
                        elif route.operation=='MODIFY_DOWN': kind=ActionKind.MODIFY_DOWN; delta=-1
                        elif route.operation=='DORMANT': kind=ActionKind.DORMANT; delta=0
                        elif route.operation=='RETIRE': kind=ActionKind.RETIRE; delta=0
                        else: kind=None; delta=0
                        if kind is not None:
                            c=ActionCandidate(route.route_id,kind,source,target,generation,evidence_ref=ev.evidence_id,rank_delta=delta)
                            reachable=self.gate.preview_any(c,ctx).formable
            self.access.set_state(route.route_id,retained=retained,reachable=reachable)

    def _route_id_for_candidate(self,c):
        if c.generation is None or c.source is None or c.target is None or c.kind is ActionKind.CREATE:
            return None
        cid=f'C:{c.source}->{c.target}:g{c.generation}'
        routes=self.access.routes_for_commitment(cid)
        if c.kind is ActionKind.REOPEN:
            return next((r.route_id for r in routes if r.operation=='REOPEN' and r.target_rank_or_handle==c.handle_id),None)
        op=c.kind.value
        if c.kind in (ActionKind.MODIFY_UP,ActionKind.MODIFY_DOWN):
            try:
                edge=self.graph.edge(EdgeIdentity(c.source,c.target,c.generation))
                target_rank=edge.active_rank+(1 if c.kind is ActionKind.MODIFY_UP else -1)
            except Exception:
                return None
            return next((r.route_id for r in routes if r.operation==op and r.target_rank_or_handle==f'r{target_rank}'),None)
        return next((r.route_id for r in routes if r.operation==op),None)

    def _feature_context(self,c,ctx):
        p=self.gate.preview_any(c,ctx)
        full=self.module_ema[c.target]['full'] if c.target else 0.0
        inq,rev=self._pressures[c.target] if c.target else (0.0,0.0)
        edge=next((e for e in ctx.edge_views if c.generation is not None and e.identity==EdgeIdentity(c.source,c.target,c.generation)),None)
        evv=next((x for x in ctx.evidence_views if x.evidence.evidence_id==c.evidence_ref),None) if c.evidence_ref else None
        ev=evv.evidence if evv else None
        h=next((x for x in ctx.handle_views if x.handle_id==c.handle_id),None) if c.handle_id else None
        from .features import future_dry_run
        prof=self.access.profile()
        dry=future_dry_run(c,self.gate,ctx,route_count=prof.route_count,retained_before=prof.retained,reachable_before=prof.reachable)
        structural=c.kind in {ActionKind.CREATE,ActionKind.MODIFY_UP,ActionKind.MODIFY_DOWN,ActionKind.DORMANT,ActionKind.RETIRE,ActionKind.REOPEN}
        predicted=self.access.preview_profile(c,structural_cooldown=structural)
        predicted_state=self.access.preview_state(c,structural_cooldown=structural)
        reachable_by_commitment={}
        for route in self.access.routes:
            if predicted_state.get(route.route_id,(False,False))[1]:
                reachable_by_commitment[route.commitment_id]=reachable_by_commitment.get(route.commitment_id,0)+1
        pair=(c.source,c.target) if c.source is not None and c.target is not None else None
        hist=self._related_history.get(pair,{}) if pair else {}
        cid=None if c.generation is None or pair is None else f'C:{c.source}->{c.target}:g{c.generation}'
        exercised=False
        if cid is not None:
            route_ids={r.route_id for r in self.access.routes_for_commitment(cid)}
            exercised=any(o.route_id in route_ids for o in self.access.outcomes)
        vh_before=sum(1 for x in ctx.handle_views if x.state.value=='VALID')
        return FeatureContext(
            step=ctx.t,preview=p,full_loss_ema=full,inquiry_pressure=inq,revision_pressure=rev,
            contribution_ema=edge.contribution_ema if edge else 0.0,usage_ema=edge.usage_ema if edge else 0.0,
            probe_gain=getattr(ev,'gain',0.0),probe_baseline_mse=getattr(ev,'baseline_mse',0.0),
            evidence_expires_step=getattr(ev,'expires_step',None),active_rank=edge.active_rank if edge else (h.active_rank if h else 0),
            probe_budget_remaining=ctx.probe_budget_remaining,edit_budget_remaining=ctx.edit_budget_remaining,
            route_count=prof.route_count,retained_before=prof.retained,reachable_before=prof.reachable,
            retained_after=predicted.retained,reachable_after=predicted.reachable,
            post_probe_remaining=dry.probe_remaining_after,post_formable_nonretire_classes=dry.formable_nonretire_class_count,
            valid_handles_before=vh_before,valid_handles_after=dry.valid_handles_after,
            total_reachable_routes_after=predicted.reachable_count,
            max_routes_one_commitment_after=max(reachable_by_commitment.values(),default=0),
            edge_age=edge.age if edge else 0,generation=c.generation or 0,dormancy_version=c.dormancy_version or 0,
            historical_support=h.historical_support if h else None,
            steps_since_related_action=None if not hist else max(0,ctx.t-int(hist.get('last_step',ctx.t))),
            accepted_related=int(hist.get('accepted',0)),rejected_related=int(hist.get('rejected',0)),
            prior_correction_exercised=exercised,
        )

    def _note_related(self,c,t,accepted=None):
        if c.source is None or c.target is None:
            return
        pair=(c.source,c.target)
        h=self._related_history.setdefault(pair,{'last_step':int(t),'accepted':0,'rejected':0})
        h['last_step']=int(t)
        if accepted is True:
            h['accepted']+=1
        elif accepted is False:
            h['rejected']+=1

    def _remember_support_context(self,decision_id,c,phi,feature_ctx):
        if c.kind not in RetirementSupportGate.ALLOWED or c.generation is None or c.target is None:
            return
        key=RetirementContextKey.from_values(c.target,float(phi[1]),float(phi[2]),float(phi[3]),feature_ctx.active_rank)
        self._retire_support_pending[str(decision_id)]=SupportRow(key,c.kind)

    def _observe_valuation_step(self,t,*,task,probe,edit,maintenance,reopen):
        before=self.valuation_ledger.matured_count
        self.valuation_ledger.observe_cost_step(t,task=task,probe=probe,edit=edit,maintenance=maintenance,reopen=reopen)
        new_rows=self.valuation_ledger.matured[before:]
        for row in new_rows:
            support=self._retire_support_pending.pop(row.decision_id,None)
            if support is not None:
                self._retire_support_rows.append(support)
        if new_rows:
            self.valuation.fit(self.valuation_ledger.matured)
        return tuple(new_rows)

    def _execute_choice(self,choice,ctx):
        c=choice.candidate
        probe=edit=reopen=0.0
        self._last_action_meta={
            'proposal_id':None,'gate_reason':None,'gate_allowed':None,
            'transaction_id':None,'evidence_id':None,
        }
        route_id=self._route_id_for_candidate(c)
        if c.kind is ActionKind.PROBE:
            ev=relation_probe(CandidateRelation(c.source,c.target),self.probe_buffer,self.probe_budget,self.registry,current_step=ctx.t)
            probe=ev.cost
            self._last_action_meta['evidence_id']=ev.evidence_id
            self._note_related(c,ctx.t,None)
        elif c.kind is not ActionKind.ABSTAIN:
            op={
                ActionKind.CREATE:Operation.CREATE,
                ActionKind.MODIFY_UP:Operation.MODIFY,
                ActionKind.MODIFY_DOWN:Operation.MODIFY,
                ActionKind.DORMANT:Operation.DORMANT,
                ActionKind.RETIRE:Operation.RETIRE,
                ActionKind.REOPEN:Operation.REOPEN,
            }[c.kind]
            proposal=StructuralProposal(
                f'P{self._decision_index:06d}',op,
                EditSpec(c.source,c.target,c.generation,c.dormancy_version,c.handle_id,c.rank_delta),
                c.evidence_ref,
            )
            self._last_action_meta['proposal_id']=proposal.proposal_id
            self._last_action_meta['evidence_id']=c.evidence_ref
            fresh=self._ctx(ctx.t)
            dec=self.gate.decide(proposal,fresh)
            self._last_action_meta['gate_reason']=dec.reason
            self._last_action_meta['gate_allowed']=bool(dec.allowed)
            self._note_related(c,ctx.t,dec.allowed)
            if dec.allowed:
                from .gate import apply_accepted_proposal
                tx=apply_accepted_proposal(self.graph,self.registry,self.structural_ledger,proposal,dec)
                self._last_action_meta['transaction_id']=tx.transaction_id
                self._cooldown=1
                if route_id is not None:
                    self.access.record_unmeasured_exercise(route_id,ctx.t)
                if c.kind is ActionKind.REOPEN:
                    reopen=dec.charged_cost
                else:
                    edit=dec.charged_cost
                self._sync_corrective_access(ctx.t)
        return probe,edit,reopen

    def step(self):
        if self.world is None: raise ScientificExecutionNotAuthorized('world not constructed without authorization')
        t=self.step_count; obs=self.world.current_observation(); self.last_preupdate_encoder_fingerprint=self.encoder_fingerprint(); lat=self.backbone.encode(obs); incoming=self.graph.route(lat); local=self.backbone.decode_local(lat); full=self.backbone.decode_full(lat,incoming); nxt=self.world.advance()
        local_terms={n:float(F.mse_loss(local[n],nxt[n]).detach()) for n in NODE_IDS}; full_terms={n:float(F.mse_loss(full[n],nxt[n]).detach()) for n in NODE_IDS}; losses=self.backbone.losses(local,full,nxt)
        active_edges=self.graph.active_edges()
        with torch.no_grad():
            message_norm={e.identity:float(e.module(lat[e.identity.source]).norm().detach()) for e in active_edges}
        incoming_norm_total={n:sum(message_norm[e.identity] for e in active_edges if e.identity.target==n) for n in NODE_IDS}
        for e in active_edges:
            masked={n:list(ms) for n,ms in incoming.items()}
            target=e.identity.target
            idx=[x.identity for x in active_edges if x.identity.target==target].index(e.identity)
            masked[target]=masked[target][:idx]+masked[target][idx+1:]
            with torch.no_grad():
                ab=self.backbone.decode_full(lat,masked)
                lmask=sum(F.mse_loss(ab[n],nxt[n]) for n in NODE_IDS)
                c=float(lmask-losses.full.detach())
                denom=incoming_norm_total[target]
                usage_share=0.0 if denom<=0.0 else message_norm[e.identity]/denom
                e.contribution_ema=self._ema(e.contribution_ema,c)
                e.usage_ema=self._ema(e.usage_ema,usage_share)
            if c>0:
                e.latest_positive_contribution=float(c)
        self.backbone_opt.zero_grad(set_to_none=True)
        for e in self.graph.active_edges():
            if e.optimizer: e.optimizer.zero_grad(set_to_none=True)
        losses.train.backward(); self.backbone_opt.step()
        for e in self.graph.active_edges():
            if e.optimizer: e.optimizer.step(); e.age+=1
        residuals={n:(nxt[n]-local[n].detach()) for n in NODE_IDS}; self.probe_buffer.append(t,lat,residuals,encoder_fingerprint=self.last_preupdate_encoder_fingerprint,target_revealed=True)
        maintenance=sum(e.active_rank*self.bob_config.maintenance_cost_per_rank for e in self.graph.active_edges())
        boundary=((t+1)%self.bob_config.structural_interval==0); self._update_pressures(local_terms,full_terms,boundary)
        probe=edit=reopen=0.0; choice=None; candidates=[]; decision_id=None; self._last_action_meta={}
        if boundary:
            self.probe_budget.reset_window()
            self._emit_edge_evidence(t)
            self._sync_corrective_access(t)
            ctx=self._ctx(t)
            candidates=self.manager.build_candidates(ctx)
            fctx={c.candidate_id:self._feature_context(c,ctx) for c in candidates}
            choice=self.manager.choose_action(candidates,fctx,decision_index=self._decision_index)
            decision_id=f'D{self._decision_index:06d}'
            chosen_phi=choice.features[choice.candidate.candidate_id]
            self.valuation_ledger.record_decision(decision_id,t,chosen_phi,chosen_by=choice.mode,candidate_id=choice.candidate.candidate_id,predicted_return=choice.scores[choice.candidate.candidate_id])
            self._remember_support_context(decision_id,choice.candidate,chosen_phi,fctx[choice.candidate.candidate_id])
            probe,edit,reopen=self._execute_choice(choice,ctx)
            self._decision_index+=1
            if self._cooldown>0 and edit==0 and reopen==0:
                self._cooldown=max(0,self._cooldown-1)
                self._sync_corrective_access(t)
        newly_matured=self._observe_valuation_step(t,task=float(losses.full.detach()),probe=probe,edit=edit,maintenance=maintenance,reopen=reopen)
        self.cost_history.append({'step':t,'task':float(losses.full.detach()),'probe':probe,'edit':edit,'maintenance':maintenance,'reopen':reopen})
        from .trajectory import structural_fingerprint, parameter_fingerprint, valuation_fingerprint
        prof=self.access.profile(); total_exposure=sum(self.manager.exposure.class_counts.values()); abst=self.manager.exposure.class_counts.get(ActionKind.ABSTAIN,0)
        graph_state={
            'edges':[{'source':e.identity.source,'target':e.identity.target,'generation':e.identity.generation,'status':e.status.value,'rank':e.active_rank} for e in self.graph.edges()],
            'handles':[{'handle_id':h.handle_id,'source':h.identity.source,'target':h.identity.target,'generation':h.identity.generation,'dormancy_version':h.dormancy_version,'state':h.state.value} for h in self.graph.handles()],
        }
        supported_now=[f'{e.identity.source}->{e.identity.target}:g{e.identity.generation}' for e in self.graph.active_edges() if self._retire_support_gate.supported(self._retirement_key(e),self._retire_support_rows)]
        record={
            'step':t,'loss':float(losses.full.detach()),'local_loss':float(losses.local.detach()),
            'probe_cost':probe,'edit_cost':edit,'maintenance_cost':maintenance,'reopen_cost':reopen,
            'probe_spend':self.probe_budget.spent_lifetime,'active_edges':len(self.graph.active_edges()),
            'retained':prof.retained,'reachable':prof.reachable,'retained_count':prof.retained_count,'reachable_count':prof.reachable_count,'route_count':prof.route_count,
            'structural_fingerprint':structural_fingerprint(self.graph),
            'parameter_fingerprint':parameter_fingerprint(self.backbone,self.graph),
            'valuation_fingerprint':valuation_fingerprint(self.valuation,self.valuation_ledger.matured_count),
            'eta':[float(x) for x in self.valuation.eta.tolist()],
            'action_exposure':{k.value:int(v) for k,v in self.manager.exposure.class_counts.items()},
            'action_identity_exposure':{str(k):int(v) for k,v in self.manager.exposure.identity_counts.items()},
            'action_eligibility':{k.value:int(v) for k,v in self.manager.exposure.eligibility_counts.items()},
            'abstain_rate':(abst/total_exposure if total_exposure else 0.0),
            'pending_decisions':len(self.valuation_ledger.pending),'matured_decisions':self.valuation_ledger.matured_count,
            'pending_decision_ids':[r.decision_id for r in self.valuation_ledger.pending],
            'matured_decision_ids':[r.decision_id for r in newly_matured],
            'matured_realized_returns':[float(r.realized_return) for r in newly_matured],
            'matured_prediction_residuals':[float(r.prediction_residual) for r in newly_matured],
            'decision_id':decision_id,
            'decision_mode':None if choice is None else choice.mode,
            'chosen_candidate':None if choice is None else choice.candidate.candidate_id,
            'eligible_candidates':[c.candidate_id for c in candidates],
            'candidate_scores':{} if choice is None else {k:float(v) for k,v in choice.scores.items()},
            'candidate_features':{} if choice is None else {k:[float(x) for x in v.tolist()] for k,v in choice.features.items()},
            'retire_support_rows':len(self._retire_support_rows),
            'retire_supported_edges':sorted(supported_now),
            'graph_state':graph_state,
            'proposal_id':self._last_action_meta.get('proposal_id'),
            'gate_reason':self._last_action_meta.get('gate_reason'),
            'gate_allowed':self._last_action_meta.get('gate_allowed'),
            'transaction_id':self._last_action_meta.get('transaction_id'),
            'evidence_id':self._last_action_meta.get('evidence_id'),
        }
        self.recorder.append(record)
        self.step_count+=1; return {'step':t,'loss':float(losses.full.detach())}
    def run_conformance_steps(self,n):
        if not self.conformance_only: raise ScientificExecutionNotAuthorized('conformance mode required')
        return [self.step() for _ in range(int(n))]
    def run_lifetime(self,authorization,repo_root='.'):
        from .authorization import verify_execution_authorization, RepoState
        try:
            repo_state=RepoState.from_git(repo_root)
            verify_execution_authorization(authorization,repo_state)
        except Exception as exc:
            raise ScientificExecutionNotAuthorized(str(exc)) from exc
        self.world=RotatingDependencyWorld(self.world_config)
        out=[]
        while self.step_count < self.world.total_steps:
            out.append(self.step())
        return out
