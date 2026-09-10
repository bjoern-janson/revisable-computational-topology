import torch
from rct.bob.config import WorldConfig
from rct.bob.world import RotatingDependencyWorld


def test_private_schedule_is_sampled_and_not_exposed_in_observation():
    w = RotatingDependencyWorld(WorldConfig())
    assert all(1536 <= n <= 2560 for n in w.phase_lengths)
    x = w.current_observation()
    assert set(x) == {"A", "B", "C", "D"}
    assert all(v.shape == (16,) for v in x.values())
    assert not hasattr(x, "regime")


def test_world_seed_reproduces_schedule_and_initial_observation():
    a = RotatingDependencyWorld(WorldConfig())
    b = RotatingDependencyWorld(WorldConfig())
    assert a.phase_lengths == b.phase_lengths
    xa, xb = a.current_observation(), b.current_observation()
    for node in xa:
        assert torch.equal(xa[node], xb[node])


def test_transition_matrices_are_orthogonal_and_pairing_changes():
    w = RotatingDependencyWorld(WorldConfig())
    r1 = w.audit_transition_matrix("W1")
    r3 = w.audit_transition_matrix("W3")
    eye = torch.eye(r1.shape[0], dtype=r1.dtype)
    assert torch.allclose(r1.T @ r1, eye, atol=1e-10)
    assert torch.allclose(r3.T @ r3, eye, atol=1e-10)
    assert not torch.allclose(r1, r3)
