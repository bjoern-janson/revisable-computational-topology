from dataclasses import fields

import rct.bob.types as bob_types


def test_probe_acquisition_warrant_is_a_distinct_typed_object():
    assert hasattr(bob_types, "ProbeAcquisitionWarrant")
    cls = bob_types.ProbeAcquisitionWarrant
    assert [f.name for f in fields(cls)] == [
        "warrant_id",
        "source",
        "target",
        "created_step",
        "expires_step",
    ]


def test_probe_warrant_registry_tracks_freshness_and_consumption():
    from rct.bob.warrant import ProbeWarrantRegistry

    w = bob_types.ProbeAcquisitionWarrant('W1', 'A', 'B', 64, 128)
    registry = ProbeWarrantRegistry()
    registry.add(w)
    assert registry.is_usable('W1', 64)
    assert registry.is_usable('W1', 128)
    assert not registry.is_usable('W1', 129)
    registry.consume('W1')
    assert not registry.is_usable('W1', 100)


def test_gate_context_carries_warrant_state_separately_from_evidence():
    import rct.bob.gate as gate

    assert hasattr(gate, 'WarrantView')
    assert 'warrant_views' in [f.name for f in fields(gate.GateContext)]


def test_probe_evidence_carries_warrant_reference_not_warrant_authority():
    assert 'warrant_ref' in [f.name for f in fields(bob_types.ProbeEvidence)]


def _handoff_ctx(*, t, inquiry_windows, evidence_views=(), warrant_views=()):
    from rct.bob.gate import GateContext, TargetPressure

    pressures = tuple(
        TargetPressure(n, 0.2, 0.0, inquiry_windows, 0)
        for n in ('A', 'B', 'C', 'D')
    )
    return GateContext(
        t=t,
        probe_budget_remaining=2,
        edit_budget_remaining=0.5,
        cooldown_remaining=0,
        pressures=pressures,
        edge_views=(),
        handle_views=(),
        evidence_views=tuple(evidence_views),
        retire_supported_edges=frozenset(),
        applied_proposal_ids=frozenset(),
        probe_rows_available=256,
        warrant_views=tuple(warrant_views),
    )


def test_fresh_probe_warrant_carries_create_eligibility_across_pressure_drop():
    from rct.bob.config import BobConfig
    from rct.bob.gate import EvidenceView, StructuralGate, WarrantView

    gate = StructuralGate(BobConfig())
    warrant = bob_types.ProbeAcquisitionWarrant('W1', 'A', 'B', 64, 128)
    evidence = bob_types.ProbeEvidence(
        'E1', 'A', 'B', 0.2, 1.0, 0.8, 0.02, 64, 128, warrant_ref='W1'
    )
    ctx = _handoff_ctx(
        t=128,
        inquiry_windows=0,
        evidence_views=(EvidenceView(evidence, False),),
        warrant_views=(WarrantView(warrant, False),),
    )
    candidate = bob_types.ActionCandidate(
        'CREATE:E1', bob_types.ActionKind.CREATE, 'A', 'B', evidence_ref='E1'
    )

    preview = gate.preview(candidate, ctx)

    assert preview.formable, preview.reason


def test_gate_issues_probe_warrant_only_after_probe_is_formable():
    from rct.bob.config import BobConfig
    from rct.bob.gate import StructuralGate
    from rct.bob.warrant import ProbeWarrantRegistry

    gate = StructuralGate(BobConfig())
    registry = ProbeWarrantRegistry()
    probe = bob_types.ActionCandidate('P', bob_types.ActionKind.PROBE, 'A', 'B')
    qualified = _handoff_ctx(t=64, inquiry_windows=3)

    warrant = gate.issue_probe_warrant(probe, qualified, registry)

    assert warrant.warrant_id == 'W000001'
    assert (warrant.source, warrant.target) == ('A', 'B')
    assert (warrant.created_step, warrant.expires_step) == (64, 128)
    assert registry.is_usable(warrant.warrant_id, 128)

    unqualified = _handoff_ctx(t=128, inquiry_windows=0)
    try:
        gate.issue_probe_warrant(
            bob_types.ActionCandidate('P2', bob_types.ActionKind.PROBE, 'A', 'C'),
            unqualified,
            registry,
        )
    except ValueError as exc:
        assert 'INQUIRY_PRESSURE_NOT_QUALIFIED' in str(exc)
    else:
        raise AssertionError('unqualified PROBE must not mint a warrant')


def test_relation_probe_requires_and_records_warrant_reference():
    import inspect
    from rct.bob.probe import relation_probe

    param = inspect.signature(relation_probe).parameters.get('warrant_ref')
    assert param is not None
    assert param.default is inspect._empty


def test_successful_create_consumes_evidence_and_acquisition_warrant():
    from rct.bob.config import BobConfig
    from rct.bob.gate import EvidenceView, StructuralGate, WarrantView, apply_accepted_proposal
    from rct.bob.interfaces import AdapterGraph
    from rct.bob.lineage import StructuralLedger
    from rct.bob.probe import EvidenceRegistry
    from rct.bob.warrant import ProbeWarrantRegistry

    gate = StructuralGate(BobConfig(hidden_dim=8, interface_max_rank=4, interface_initial_rank=2))
    graph = AdapterGraph(8, 4, 2)
    evidence_registry = EvidenceRegistry()
    warrant_registry = ProbeWarrantRegistry()
    warrant = bob_types.ProbeAcquisitionWarrant('W1', 'A', 'B', 64, 128)
    warrant_registry.add(warrant)
    evidence = bob_types.ProbeEvidence(
        'E1', 'A', 'B', 0.2, 1.0, 0.8, 0.02, 64, 128, warrant_ref='W1'
    )
    evidence_registry.add(evidence)
    ctx = _handoff_ctx(
        t=128,
        inquiry_windows=0,
        evidence_views=(EvidenceView(evidence, False),),
        warrant_views=(WarrantView(warrant, False),),
    )
    proposal = bob_types.StructuralProposal(
        'P1', bob_types.Operation.CREATE, bob_types.EditSpec('A', 'B'), 'E1'
    )
    decision = gate.decide(proposal, ctx)
    assert decision.allowed

    apply_accepted_proposal(
        graph,
        evidence_registry,
        StructuralLedger(),
        proposal,
        decision,
        warrant_registry=warrant_registry,
    )

    assert evidence_registry.consumed('E1')
    assert warrant_registry.consumed('W1')


def test_lifetime_probe_mints_linked_warrant_and_next_boundary_create_can_form():
    from types import SimpleNamespace
    import torch
    from rct.bob.lifetime import BobLifetime

    bob = BobLifetime(conformance_only=True)
    assert hasattr(bob, 'warrants')

    g = torch.Generator().manual_seed(123)
    for t in range(256):
        lat = {n: torch.randn(bob.bob_config.hidden_dim, generator=g) for n in ('A','B','C','D')}
        res = {n: torch.randn(bob.world_config.obs_dim, generator=g) for n in ('A','B','C','D')}
        bob.probe_buffer.append(t, lat, res, encoder_fingerprint='enc', target_revealed=True)

    bob._inq_q['B'] = 3
    bob._pressures['B'] = (0.2, 0.0)
    ctx = bob._ctx(64)
    probe = bob_types.ActionCandidate('PROBE:A->B', bob_types.ActionKind.PROBE, 'A', 'B')
    assert bob.gate.preview(probe, ctx).formable

    probe_cost, edit_cost, reopen_cost = bob._execute_choice(SimpleNamespace(candidate=probe), ctx)
    assert probe_cost == bob.bob_config.probe_cost
    assert edit_cost == 0.0 and reopen_cost == 0.0
    evidence = bob.registry.items()[-1]
    assert evidence.warrant_ref is not None
    assert bob.warrants.is_usable(evidence.warrant_ref, 128)

    bob._inq_q['B'] = 0
    bob._pressures['B'] = (0.0, 0.0)
    next_ctx = bob._ctx(128)
    candidates = bob.manager.build_candidates(next_ctx)
    creates = [c for c in candidates if c.kind is bob_types.ActionKind.CREATE and c.evidence_ref == evidence.evidence_id]
    assert len(creates) == 1
    assert bob.gate.preview(creates[0], next_ctx).formable
