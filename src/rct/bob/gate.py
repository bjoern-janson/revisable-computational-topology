from __future__ import annotations

from dataclasses import dataclass

from .config import BobConfig
from .interfaces import AdapterGraph
from .lineage import EdgeIdentity, StructuralLedger, StructuralTransaction
from .probe import EvidenceRegistry
from .types import (
    ActionCandidate, ActionKind, EdgeStatus, ExistingEdgeEvidence, GateDecision,
    HandleState, NodeId, Operation, ProbeEvidence, ReopenEvidence, StructuralProposal,
)


@dataclass(frozen=True)
class TargetPressure:
    target: NodeId
    inquiry_pressure: float
    revision_pressure: float
    inquiry_qualifying_windows: int
    revision_qualifying_windows: int


@dataclass(frozen=True)
class EdgeView:
    identity: EdgeIdentity
    status: EdgeStatus
    active_rank: int
    max_rank: int
    age: int = 0
    usage_ema: float = 0.0
    contribution_ema: float = 0.0


@dataclass(frozen=True)
class HandleView:
    handle_id: str
    identity: EdgeIdentity
    dormancy_version: int
    state: HandleState
    historical_support: float | None
    reopening_liability: float
    active_rank: int


@dataclass(frozen=True)
class EvidenceView:
    evidence: ProbeEvidence | ExistingEdgeEvidence | ReopenEvidence
    consumed: bool


@dataclass(frozen=True)
class GateContext:
    t: int
    probe_budget_remaining: int
    edit_budget_remaining: float
    cooldown_remaining: int
    pressures: tuple[TargetPressure, ...]
    edge_views: tuple[EdgeView, ...]
    handle_views: tuple[HandleView, ...]
    evidence_views: tuple[EvidenceView, ...]
    retire_supported_edges: frozenset[EdgeIdentity]
    applied_proposal_ids: frozenset[str]
    probe_rows_available: int


@dataclass(frozen=True)
class EligibilityPreview:
    candidate: ActionCandidate
    formable: bool
    reason: str
    probe_charge: float = 0.0
    edit_charge: float = 0.0
    reopen_liability: float = 0.0
    maintenance_delta: float = 0.0

    @property
    def total_immediate_charge(self) -> float:
        return self.probe_charge + self.edit_charge + self.reopen_liability


class StructuralGate:
    def __init__(self, config: BobConfig):
        self.config = config

    def _pressure(self, ctx: GateContext, target: NodeId | None) -> TargetPressure | None:
        if target is None:
            return None
        return next((p for p in ctx.pressures if p.target == target), None)

    def _edge(self, ctx: GateContext, c: ActionCandidate) -> EdgeView | None:
        if c.source is None or c.target is None or c.generation is None:
            return None
        ident = EdgeIdentity(c.source, c.target, c.generation)
        return next((e for e in ctx.edge_views if e.identity == ident), None)

    def _handle(self, ctx: GateContext, c: ActionCandidate) -> HandleView | None:
        if c.handle_id is None:
            return None
        return next((h for h in ctx.handle_views if h.handle_id == c.handle_id), None)

    def _evidence(self, ctx: GateContext, evidence_ref: str | None) -> EvidenceView | None:
        if evidence_ref is None:
            return None
        return next((e for e in ctx.evidence_views if e.evidence.evidence_id == evidence_ref), None)

    def _usable_evidence(self, ctx: GateContext, evidence_ref: str | None) -> EvidenceView | None:
        view = self._evidence(ctx, evidence_ref)
        if view is None or view.consumed:
            return None
        ev = view.evidence
        if not (ev.created_step <= ctx.t <= ev.expires_step):
            return None
        return view

    def _duplicate_active_endpoint(self, ctx: GateContext, source: NodeId, target: NodeId) -> bool:
        return any(
            e.identity.source == source and e.identity.target == target and e.status is EdgeStatus.ACTIVE
            for e in ctx.edge_views
        )

    def _not_formable(self, c: ActionCandidate, reason: str) -> EligibilityPreview:
        return EligibilityPreview(c, False, reason)

    def preview(self, c: ActionCandidate, ctx: GateContext) -> EligibilityPreview:
        cfg = self.config
        if c.kind is ActionKind.ABSTAIN:
            return EligibilityPreview(c, True, "ABSTAIN_AVAILABLE")
        if c.source is None or c.target is None or c.source == c.target:
            return self._not_formable(c, "ILLEGAL_ENDPOINTS")
        pressure = self._pressure(ctx, c.target)
        if pressure is None:
            return self._not_formable(c, "MISSING_TARGET_PRESSURE")
        if c.kind is ActionKind.PROBE:
            if pressure.inquiry_qualifying_windows < 3:
                return self._not_formable(c, "INQUIRY_PRESSURE_NOT_QUALIFIED")
            if ctx.probe_rows_available < cfg.probe_window:
                return self._not_formable(c, "PROBE_DATA_NOT_READY")
            if ctx.probe_budget_remaining <= 0:
                return self._not_formable(c, "PROBE_BUDGET_EXHAUSTED")
            if self._duplicate_active_endpoint(ctx, c.source, c.target):
                return self._not_formable(c, "RELATION_ALREADY_ACTIVE")
            return EligibilityPreview(c, True, "FORMABLE", probe_charge=cfg.probe_cost)
        if ctx.cooldown_remaining > 0:
            return self._not_formable(c, "STRUCTURAL_COOLDOWN")
        ev_view = self._usable_evidence(ctx, c.evidence_ref)
        if ev_view is None:
            return self._not_formable(c, "MISSING_STALE_OR_CONSUMED_EVIDENCE")
        ev = ev_view.evidence
        if c.kind is ActionKind.CREATE:
            if pressure.inquiry_qualifying_windows < 3:
                return self._not_formable(c, "INQUIRY_PRESSURE_NOT_QUALIFIED")
            if not isinstance(ev, ProbeEvidence):
                return self._not_formable(c, "WRONG_EVIDENCE_TYPE")
            if (ev.source, ev.target) != (c.source, c.target):
                return self._not_formable(c, "EVIDENCE_ENDPOINT_MISMATCH")
            if self._duplicate_active_endpoint(ctx, c.source, c.target):
                return self._not_formable(c, "DUPLICATE_ACTIVE_ENDPOINT")
            charge = cfg.create_cost
            if charge > ctx.edit_budget_remaining:
                return self._not_formable(c, "EDIT_BUDGET_EXHAUSTED")
            return EligibilityPreview(c, True, "FORMABLE", edit_charge=charge,
                                      maintenance_delta=cfg.maintenance_cost_per_rank * cfg.interface_initial_rank)
        edge = self._edge(ctx, c)
        if edge is None or edge.status is not EdgeStatus.ACTIVE:
            return self._not_formable(c, "MISSING_ACTIVE_EDGE")
        if pressure.revision_qualifying_windows < 3:
            return self._not_formable(c, "REVISION_PRESSURE_NOT_QUALIFIED")
        if not isinstance(ev, ExistingEdgeEvidence) and c.kind is not ActionKind.REOPEN:
            return self._not_formable(c, "WRONG_EVIDENCE_TYPE")
        if isinstance(ev, ExistingEdgeEvidence):
            if (ev.source, ev.target, ev.generation) != (c.source, c.target, c.generation):
                return self._not_formable(c, "EVIDENCE_IDENTITY_MISMATCH")
        if c.kind in (ActionKind.MODIFY_UP, ActionKind.MODIFY_DOWN):
            delta = 1 if c.kind is ActionKind.MODIFY_UP else -1
            if c.rank_delta not in (0, delta):
                return self._not_formable(c, "RANK_DELTA_MISMATCH")
            if not 1 <= edge.active_rank + delta <= edge.max_rank:
                return self._not_formable(c, "RANK_BOUND")
            if cfg.modify_cost > ctx.edit_budget_remaining:
                return self._not_formable(c, "EDIT_BUDGET_EXHAUSTED")
            return EligibilityPreview(c, True, "FORMABLE", edit_charge=cfg.modify_cost,
                                      maintenance_delta=delta * cfg.maintenance_cost_per_rank)
        if c.kind is ActionKind.DORMANT:
            if cfg.dormant_cost > ctx.edit_budget_remaining:
                return self._not_formable(c, "EDIT_BUDGET_EXHAUSTED")
            return EligibilityPreview(c, True, "FORMABLE", edit_charge=cfg.dormant_cost,
                                      maintenance_delta=-edge.active_rank * cfg.maintenance_cost_per_rank)
        if c.kind is ActionKind.RETIRE:
            if edge.identity not in ctx.retire_supported_edges:
                return self._not_formable(c, "RETIRE_SUPPORT_REQUIRED")
            if cfg.retire_cost > ctx.edit_budget_remaining:
                return self._not_formable(c, "EDIT_BUDGET_EXHAUSTED")
            return EligibilityPreview(c, True, "FORMABLE", edit_charge=cfg.retire_cost,
                                      maintenance_delta=-edge.active_rank * cfg.maintenance_cost_per_rank)
        if c.kind is ActionKind.REOPEN:
            return self._not_formable(c, "REOPEN_REQUIRES_HANDLE_PATH")
        return self._not_formable(c, "UNKNOWN_ACTION")

    def preview_reopen(self, c: ActionCandidate, ctx: GateContext) -> EligibilityPreview:
        if c.kind is not ActionKind.REOPEN:
            return self.preview(c, ctx)
        if c.source is None or c.target is None or c.generation is None:
            return self._not_formable(c, "MISSING_REOPEN_IDENTITY")
        if ctx.cooldown_remaining > 0:
            return self._not_formable(c, "STRUCTURAL_COOLDOWN")
        pressure = self._pressure(ctx, c.target)
        if pressure is None or pressure.revision_qualifying_windows < 3:
            return self._not_formable(c, "REVISION_PRESSURE_NOT_QUALIFIED")
        if self._duplicate_active_endpoint(ctx, c.source, c.target):
            return self._not_formable(c, "DUPLICATE_ACTIVE_ENDPOINT")
        handle = self._handle(ctx, c)
        if handle is None or handle.state is not HandleState.VALID:
            return self._not_formable(c, "INVALID_HANDLE")
        if handle.identity != EdgeIdentity(c.source, c.target, c.generation) or handle.dormancy_version != c.dormancy_version:
            return self._not_formable(c, "HANDLE_IDENTITY_MISMATCH")
        if handle.historical_support is None or handle.historical_support <= 0:
            return self._not_formable(c, "NO_HISTORICAL_SUPPORT")
        ev_view = self._usable_evidence(ctx, c.evidence_ref)
        if ev_view is None or not isinstance(ev_view.evidence, ReopenEvidence):
            return self._not_formable(c, "MISSING_OR_WRONG_REOPEN_EVIDENCE")
        ev = ev_view.evidence
        if (ev.source, ev.target, ev.generation, ev.dormancy_version, ev.handle_id) != (c.source, c.target, c.generation, c.dormancy_version, c.handle_id):
            return self._not_formable(c, "REOPEN_EVIDENCE_IDENTITY_MISMATCH")
        liability = float(handle.reopening_liability)
        if liability > ctx.edit_budget_remaining:
            return self._not_formable(c, "REOPEN_BUDGET_EXHAUSTED")
        return EligibilityPreview(c, True, "FORMABLE", reopen_liability=liability,
                                  maintenance_delta=handle.active_rank * self.config.maintenance_cost_per_rank)

    def preview_any(self, c: ActionCandidate, ctx: GateContext) -> EligibilityPreview:
        return self.preview_reopen(c, ctx) if c.kind is ActionKind.REOPEN else self.preview(c, ctx)

    def decide(self, proposal: StructuralProposal, ctx: GateContext) -> GateDecision:
        if proposal.proposal_id in ctx.applied_proposal_ids:
            return GateDecision(proposal.proposal_id, False, "PROPOSAL_ALREADY_APPLIED", 0.0)
        edit = proposal.edit
        kind = {
            Operation.CREATE: ActionKind.CREATE,
            Operation.DORMANT: ActionKind.DORMANT,
            Operation.RETIRE: ActionKind.RETIRE,
            Operation.REOPEN: ActionKind.REOPEN,
        }.get(proposal.operation)
        if proposal.operation is Operation.MODIFY:
            kind = ActionKind.MODIFY_UP if edit.rank_delta > 0 else ActionKind.MODIFY_DOWN
        if kind is None:
            return GateDecision(proposal.proposal_id, False, "UNKNOWN_OPERATION", 0.0)
        c = ActionCandidate(proposal.proposal_id, kind, edit.source, edit.target, edit.generation,
                            edit.dormancy_version, edit.handle_id, proposal.evidence_ref, edit.rank_delta)
        p = self.preview_any(c, ctx)
        return GateDecision(proposal.proposal_id, p.formable, p.reason,
                            p.total_immediate_charge if p.formable else 0.0)


def apply_accepted_proposal(graph: AdapterGraph, registry: EvidenceRegistry, ledger: StructuralLedger,
                            proposal: StructuralProposal, decision: GateDecision) -> StructuralTransaction:
    if not decision.allowed:
        raise ValueError("cannot apply rejected proposal")
    if ledger.proposal_applied(proposal.proposal_id):
        raise ValueError("proposal already applied")
    edit = proposal.edit
    before = graph.structural_fingerprint()
    reopen_handle_id = None
    if proposal.operation is Operation.CREATE:
        if edit.source is None or edit.target is None: raise ValueError("CREATE missing endpoints")
        graph.create(edit.source, edit.target)
    elif proposal.operation is Operation.MODIFY:
        if edit.source is None or edit.target is None or edit.generation is None: raise ValueError("MODIFY missing identity")
        graph.modify_rank(EdgeIdentity(edit.source, edit.target, edit.generation), edit.rank_delta)
    elif proposal.operation is Operation.DORMANT:
        if edit.source is None or edit.target is None or edit.generation is None: raise ValueError("DORMANT missing identity")
        h = graph.dormant(EdgeIdentity(edit.source, edit.target, edit.generation)); reopen_handle_id = h.handle_id
    elif proposal.operation is Operation.RETIRE:
        if edit.source is None or edit.target is None or edit.generation is None: raise ValueError("RETIRE missing identity")
        graph.retire(EdgeIdentity(edit.source, edit.target, edit.generation))
    elif proposal.operation is Operation.REOPEN:
        if edit.handle_id is None: raise ValueError("REOPEN missing handle")
        graph.reopen(edit.handle_id); reopen_handle_id = edit.handle_id
    else:
        raise ValueError("unknown operation")
    if proposal.evidence_ref is not None:
        registry.consume(proposal.evidence_ref)
    tx = StructuralTransaction(
        transaction_id=f"T{len(ledger.transactions) + 1:06d}", proposal_id=proposal.proposal_id,
        operation=proposal.operation, graph_before=before, graph_after=graph.structural_fingerprint(),
        evidence_ref=proposal.evidence_ref, affected_routes=proposal.affected_routes,
        reopen_handle_id=reopen_handle_id, charged_cost=decision.charged_cost,
    )
    ledger.append_transaction(tx)
    return tx
