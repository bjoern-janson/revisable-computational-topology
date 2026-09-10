from dataclasses import FrozenInstanceError
import pytest

from rct.bob.config import BobConfig, WorldConfig, MkIIConfig
from rct.bob.types import (
    NODE_IDS, Operation, ActionKind, HandleState,
    ActionCandidate, ProbeEvidence, ExistingEdgeEvidence, ReopenEvidence,
    ReopenLiability,
)


def test_exact_vocabularies_and_mkii_constants():
    assert NODE_IDS == ("A", "B", "C", "D")
    assert {x.value for x in Operation} == {"CREATE", "MODIFY", "DORMANT", "RETIRE", "REOPEN"}
    assert {x.value for x in ActionKind} == {
        "PROBE", "CREATE", "MODIFY_UP", "MODIFY_DOWN",
        "DORMANT", "RETIRE", "REOPEN", "ABSTAIN",
    }
    assert {x.value for x in HandleState} == {"VALID", "CONSUMED", "REVOKED"}
    m = MkIIConfig()
    assert m.return_horizon_steps == 256
    assert m.eta_ridge_lambda == 1.0
    assert m.feature_dim == 38
    assert m.exploration_block_size == 4
    assert m.exploration_slots_per_block == 1


def test_world_and_bob_defaults_are_frozen_contracts():
    b = BobConfig()
    w = WorldConfig()
    assert b.structural_interval == 64
    assert b.probe_window == 256
    assert b.probes_per_window == 2
    assert b.edit_budget_per_window == pytest.approx(0.50)
    assert (w.alpha, w.beta, w.rho) == pytest.approx((0.6, 0.8, 0.85))
    assert (w.w1_steps_min, w.w1_steps_max) == (1536, 2560)
    with pytest.raises(FrozenInstanceError):
        b.hidden_dim = 99


def test_action_and_evidence_records_are_immutable_and_typed():
    c = ActionCandidate("C1", ActionKind.CREATE, source="A", target="B", evidence_ref="E1")
    assert c.source == "A" and c.target == "B"
    p = ProbeEvidence("E1", "A", "B", gain=0.2, baseline_mse=1.0, probed_mse=0.8,
                      cost=0.02, created_step=64, expires_step=128)
    e = ExistingEdgeEvidence("E2", "A", "B", 1, revision_pressure=0.3,
                             contribution_ema=0.1, usage_ema=0.5, active_rank=2,
                             created_step=64, expires_step=128)
    r = ReopenEvidence("E3", "A", "B", 1, 2, "H1", current_pressure=0.4,
                       historical_support=0.2, reopening_liability=0.1,
                       created_step=64, expires_step=128)
    assert (p.evidence_id, e.generation, r.dormancy_version) == ("E1", 1, 2)
    with pytest.raises(FrozenInstanceError):
        p.gain = 3.0


def test_reopen_liability_is_additive():
    x = ReopenLiability(info=0.1, reach=0.2, effective_use=0.3)
    assert x.total == pytest.approx(0.6)
