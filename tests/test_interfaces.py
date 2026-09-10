import pytest
import torch

from rct.bob.interfaces import AdapterGraph, EdgeIdentity, InvalidHandle
from rct.bob.types import EdgeStatus, HandleState


def make_graph():
    return AdapterGraph(hidden_dim=8, max_rank=4, initial_rank=2, edge_lr=1e-3)


def test_create_is_directional_and_generational():
    g = make_graph()
    e1 = g.create("A", "B")
    assert e1.identity == EdgeIdentity("A", "B", 1)
    assert g.is_active(e1.identity)
    assert not g.has_active_endpoint("B", "A")
    g.retire(e1.identity)
    e2 = g.create("A", "B")
    assert e2.identity == EdgeIdentity("A", "B", 2)


def test_active_epoch_records_latest_strictly_positive_only():
    g = make_graph()
    e = g.create("A", "B")
    g.observe_contribution(e.identity, 0.2)
    g.observe_contribution(e.identity, -1.0)
    g.observe_contribution(e.identity, 0.0)
    g.observe_contribution(e.identity, 0.1)
    assert g.edge(e.identity).latest_positive_contribution == pytest.approx(0.1)
    h = g.dormant(e.identity)
    assert h.historical_support == pytest.approx(0.1)
    g.reopen(h.handle_id)
    assert g.edge(e.identity).latest_positive_contribution is None


def test_consumed_handle_cannot_reopen_after_new_dormancy():
    g = make_graph()
    e = g.create("A", "B")
    g.observe_contribution(e.identity, 0.2)
    h1 = g.dormant(e.identity)
    g.reopen(h1.handle_id)
    g.synthetic_edge_step(e.identity)
    g.observe_contribution(e.identity, 0.1)
    h2 = g.dormant(e.identity)
    before = g.parameter_fingerprint()
    with pytest.raises(InvalidHandle):
        g.reopen(h1.handle_id)
    assert g.parameter_fingerprint() == before
    g.reopen(h2.handle_id)
    assert g.handle(h1.handle_id).state is HandleState.CONSUMED


def test_retire_revokes_valid_handles_and_destroys_live_optimizer():
    g = make_graph()
    e = g.create("A", "B")
    h = g.dormant(e.identity, historical_support=0.2)
    g.reopen(h.handle_id)
    h2 = g.dormant(e.identity, historical_support=0.3)
    g.reopen(h2.handle_id)
    g.retire(e.identity)
    assert g.edge(e.identity).status is EdgeStatus.RETIRED
    assert g.edge(e.identity).optimizer is None
    assert all(x.state is not HandleState.VALID for x in g.handles_for_generation(e.identity))


def test_rank_down_then_other_updates_do_not_drift_inactive_factors_or_moments():
    g = make_graph()
    e = g.create("A", "B")
    g.synthetic_edge_step(e.identity)
    g.modify_rank(e.identity, +1)
    g.synthetic_edge_step(e.identity)
    g.modify_rank(e.identity, -1)
    edge = g.edge(e.identity)
    r = edge.active_rank
    u_tail = edge.module.u[:, r:].detach().clone()
    v_tail = edge.module.v[r:, :].detach().clone()
    state_before = g.optimizer_fingerprint(e.identity, inactive_from=r)
    for _ in range(3):
        g.synthetic_edge_step(e.identity)
    assert torch.equal(u_tail, edge.module.u[:, r:].detach())
    assert torch.equal(v_tail, edge.module.v[r:, :].detach())
    assert state_before == g.optimizer_fingerprint(e.identity, inactive_from=r)


def test_route_uses_only_active_directional_edges():
    g = make_graph()
    e = g.create("A", "B")
    latents = {n: torch.randn(8) for n in ("A", "B", "C", "D")}
    routed = g.route(latents)
    assert len(routed["B"]) == 1
    assert routed["A"] == []
    g.dormant(e.identity, historical_support=None)
    assert g.route(latents)["B"] == []
