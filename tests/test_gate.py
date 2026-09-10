import dataclasses
import pytest

from rct.bob.config import BobConfig
from rct.bob.gate import StructuralGate, GateContext, TargetPressure, EdgeView, EvidenceView, apply_accepted_proposal
from rct.bob.interfaces import AdapterGraph, EdgeIdentity
from rct.bob.lineage import StructuralLedger
from rct.bob.probe import EvidenceRegistry
from rct.bob.types import ActionCandidate, ActionKind, ExistingEdgeEvidence, ProbeEvidence, StructuralProposal, EditSpec, Operation, EdgeStatus


def pressures(inquiry=3, revision=3):
    return tuple(TargetPressure(n, 0.2, 0.2, inquiry, revision) for n in ("A", "B", "C", "D"))


def base_ctx(*, edge_views=(), evidence_views=(), cooldown=0, probe_remaining=2, edit_remaining=0.5, retire_supported=frozenset(), applied=frozenset()):
    return GateContext(t=64, probe_budget_remaining=probe_remaining, edit_budget_remaining=edit_remaining,
                       cooldown_remaining=cooldown, pressures=pressures(), edge_views=tuple(edge_views),
                       handle_views=(), evidence_views=tuple(evidence_views), retire_supported_edges=frozenset(retire_supported),
                       applied_proposal_ids=frozenset(applied))


def test_high_value_cannot_bypass_missing_evidence():
    gate = StructuralGate(BobConfig())
    c = ActionCandidate("C1", ActionKind.CREATE, source="A", target="B", evidence_ref=None)
    preview = gate.preview(c, base_ctx())
    assert not preview.formable
    assert "evidence" in preview.reason.lower()


def test_gate_does_not_receive_hidden_world_fields():
    names = {f.name for f in dataclasses.fields(GateContext)}
    assert names.isdisjoint({"regime", "pairing", "transition_operator", "correct_edges", "future_loss"})


def test_probe_and_abstain_survive_structural_cooldown():
    gate = StructuralGate(BobConfig())
    ctx = base_ctx(cooldown=1)
    assert gate.preview(ActionCandidate("P1", ActionKind.PROBE, source="A", target="B"), ctx).formable
    assert gate.preview(ActionCandidate("A0", ActionKind.ABSTAIN), ctx).formable


def test_create_requires_fresh_matching_typed_probe_evidence():
    gate = StructuralGate(BobConfig())
    ev = ProbeEvidence("E1", "A", "B", .2, 1., .8, .02, 64, 128)
    ctx = base_ctx(evidence_views=(EvidenceView(ev, False),))
    assert gate.preview(ActionCandidate("C1", ActionKind.CREATE, source="A", target="B", evidence_ref="E1"), ctx).formable
    assert not gate.preview(ActionCandidate("C2", ActionKind.CREATE, source="A", target="C", evidence_ref="E1"), ctx).formable


def test_retire_requires_support_even_when_other_legality_holds():
    gate = StructuralGate(BobConfig())
    ident = EdgeIdentity("A", "B", 1)
    edge = EdgeView(ident, EdgeStatus.ACTIVE, 2, 8, age=100, usage_ema=.5, contribution_ema=-.2)
    ev = ExistingEdgeEvidence("E2", "A", "B", 1, .3, -.2, .5, 2, 64, 128)
    c = ActionCandidate("R1", ActionKind.RETIRE, "A", "B", 1, evidence_ref="E2")
    ctx = base_ctx(edge_views=(edge,), evidence_views=(EvidenceView(ev, False),))
    assert not gate.preview(c, ctx).formable
    supported = dataclasses.replace(ctx, retire_supported_edges=frozenset({ident}))
    assert gate.preview(c, supported).formable


def test_exact_once_transaction_consumes_evidence_and_mutates_once():
    cfg = BobConfig(hidden_dim=8, interface_max_rank=4, interface_initial_rank=2)
    gate = StructuralGate(cfg); graph = AdapterGraph(8, 4, 2); registry = EvidenceRegistry(); ledger = StructuralLedger()
    ev = ProbeEvidence("E1", "A", "B", .2, 1., .8, .02, 64, 128); registry.add(ev)
    ctx = base_ctx(evidence_views=(EvidenceView(ev, False),))
    proposal = StructuralProposal("P1", Operation.CREATE, EditSpec("A", "B"), "E1")
    decision = gate.decide(proposal, ctx); assert decision.allowed
    tx = apply_accepted_proposal(graph, registry, ledger, proposal, decision)
    assert graph.has_active_endpoint("A", "B") and registry.consumed("E1")
    before = graph.structural_fingerprint()
    with pytest.raises(ValueError): apply_accepted_proposal(graph, registry, ledger, proposal, decision)
    assert graph.structural_fingerprint() == before and ledger.transactions == (tx,)
