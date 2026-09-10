import pytest
import torch

from rct.bob.probe import (
    ProbeBudget, ProbeBudgetExhausted, LatentResidualBuffer,
    EvidenceRegistry, relation_probe,
)
from rct.bob.types import CandidateRelation, ProbeEvidence


def make_buffer():
    g = torch.Generator().manual_seed(5)
    buf = LatentResidualBuffer(maxlen=256)
    for t in range(256):
        a = torch.randn(8, generator=g)
        target_residual = 0.8 * a[:4] + 0.03 * torch.randn(4, generator=g)
        lat = {
            "A": a,
            "B": torch.randn(8, generator=g),
            "C": torch.randn(8, generator=g),
            "D": torch.randn(8, generator=g),
        }
        residuals = {
            "A": torch.randn(4, generator=g),
            "B": target_residual,
            "C": torch.randn(4, generator=g),
            "D": torch.randn(4, generator=g),
        }
        buf.append(t, lat, residuals, encoder_fingerprint=f"enc-{t//64}", target_revealed=True)
    return buf


def test_probe_budget_resets_window_not_lifetime():
    b = ProbeBudget(max_probes=2, cost_per_probe=0.02)
    b.charge(); b.charge()
    with pytest.raises(ProbeBudgetExhausted):
        b.charge()
    assert b.spent_lifetime == pytest.approx(0.04)
    b.reset_window()
    assert b.remaining == 2
    assert b.spent_lifetime == pytest.approx(0.04)


def test_relation_probe_is_pair_specific_useful_and_nonpersistent():
    buf = make_buffer()
    budget = ProbeBudget(max_probes=2, cost_per_probe=0.02)
    registry = EvidenceRegistry()
    ev = relation_probe(CandidateRelation("A", "B"), buf, budget, registry, current_step=256, warrant_ref="W000001")
    assert isinstance(ev, ProbeEvidence)
    assert (ev.source, ev.target) == ("A", "B")
    assert ev.gain > 0.1
    assert ev.expires_step == 320
    assert ev.warrant_ref == "W000001"
    assert budget.remaining == 1
    assert not hasattr(ev, "model")
    assert registry.get(ev.evidence_id) == ev


def test_evidence_expiry_and_consumption_are_distinct_from_history():
    reg = EvidenceRegistry()
    ev = ProbeEvidence("E000001", "A", "B", 0.1, 1.0, 0.9, 0.02, 64, 128)
    reg.add(ev)
    assert reg.is_usable("E000001", current_step=128)
    assert not reg.is_usable("E000001", current_step=129)
    reg.consume("E000001")
    assert not reg.is_usable("E000001", current_step=100)
    assert reg.get("E000001") == ev


def test_buffer_rejects_unrevealed_target_rows():
    buf = LatentResidualBuffer(maxlen=256)
    lat = {n: torch.zeros(8) for n in ("A", "B", "C", "D")}
    res = {n: torch.zeros(4) for n in lat}
    with pytest.raises(ValueError):
        buf.append(0, lat, res, encoder_fingerprint="x", target_revealed=False)
