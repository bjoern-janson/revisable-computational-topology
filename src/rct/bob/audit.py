from __future__ import annotations

from dataclasses import dataclass
import torch

from .config import BobConfig, WorldConfig
from .types import NODE_IDS
from .world import RotatingDependencyWorld


@dataclass(frozen=True)
class WorldAuditReport:
    analytic_latent_marginal_invariance: bool
    max_abs_mean_shift: float
    max_abs_cov_shift: float
    analytic_cross_lag_change: float
    empirical_cross_lag_change: float
    analytic_self_lag_norm: float
    analytic_partner_lag_norm: float
    empirical_self_lag_change: float
    empirical_partner_lag_change: float
    min_partner_prediction_gain: float
    block_count: int
    block_size: int
    samples_per_regime: int
    scope: str = "ENVIRONMENT_OPPORTUNITY_ONLY"
    bob_executed: bool = False


def _cov(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    x = x.to(torch.float64)
    y = x if y is None else y.to(torch.float64)
    xc = x - x.mean(0, keepdim=True)
    yc = y - y.mean(0, keepdim=True)
    return xc.T @ yc / max(1, x.shape[0] - 1)


def _ridge_predict(train_x, train_y, test_x, ridge=1e-3):
    tx = train_x.to(torch.float64)
    ty = train_y.to(torch.float64)
    vx = test_x.to(torch.float64)
    tx = torch.cat([tx, torch.ones((tx.shape[0], 1), dtype=torch.float64)], dim=1)
    vx = torch.cat([vx, torch.ones((vx.shape[0], 1), dtype=torch.float64)], dim=1)
    eye = torch.eye(tx.shape[1], dtype=torch.float64)
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(tx.T @ tx + ridge * eye, tx.T @ ty)
    return vx @ w


def _partner_map(world: RotatingDependencyWorld, regime: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for a, b in world.audit_pairing(regime):
        i, j = NODE_IDS.index(a), NODE_IDS.index(b)
        out[i] = j
        out[j] = i
    return out


def audit_world(
    world_config: WorldConfig | None = None,
    bob_config: BobConfig | None = None,
    *,
    samples: int = 4096,
) -> WorldAuditReport:
    wc = world_config or WorldConfig()
    _ = bob_config or BobConfig()
    if samples != 4096 and samples < 32:
        raise ValueError("audit samples must be >= 32")
    world = RotatingDependencyWorld(wc)
    r1 = world.audit_transition_matrix("W1")
    r3 = world.audit_transition_matrix("W3")
    eye = torch.eye(r1.shape[0], dtype=torch.float64)
    invariant = bool(
        torch.allclose(r1.T @ r1, eye, atol=1e-10, rtol=1e-10)
        and torch.allclose(r3.T @ r3, eye, atol=1e-10, rtol=1e-10)
    )
    analytic_cross = float((wc.rho * (r1 - r3)).norm().item())
    analytic_self = float((wc.rho * wc.alpha * world._q).norm().item())
    analytic_partner = float((wc.rho * wc.beta * world._q).norm().item())

    x1, y1 = world.audit_forced_sequence("W1", samples, wc.audit_seed + 10)
    x3, y3 = world.audit_forced_sequence("W3", samples, wc.audit_seed + 20)

    mean_shift = 0.0
    cov_shift = 0.0
    for j in range(4):
        mean_shift = max(mean_shift, float((x1[:, j].mean(0) - x3[:, j].mean(0)).abs().max().item()))
        cov_shift = max(cov_shift, float((_cov(x1[:, j]) - _cov(x3[:, j])).abs().max().item()))

    cross_sq = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            delta = _cov(x1[:, i], y1[:, j]) - _cov(x3[:, i], y3[:, j])
            cross_sq += float((delta * delta).sum().item())
    empirical_cross = cross_sq**0.5

    self_changes = []
    partner_changes = []
    p1 = _partner_map(world, "W1")
    p3 = _partner_map(world, "W3")
    for j in range(4):
        self_changes.append(float((_cov(x1[:, j], y1[:, j]) - _cov(x3[:, j], y3[:, j])).norm().item()))
        partner_changes.append(float((_cov(x1[:, p1[j]], y1[:, j]) - _cov(x3[:, p3[j]], y3[:, j])).norm().item()))

    n_train = 3072 if samples == 4096 else int(samples * 0.75)
    gains = []
    for regime, xx, yy in (("W1", x1, y1), ("W3", x3, y3)):
        partners = _partner_map(world, regime)
        for target in range(4):
            partner = partners[target]
            local_train = xx[:n_train, target]
            partner_train = torch.cat([xx[:n_train, target], xx[:n_train, partner]], dim=1)
            target_train = yy[:n_train, target]
            local_test = xx[n_train:, target]
            partner_test = torch.cat([xx[n_train:, target], xx[n_train:, partner]], dim=1)
            target_test = yy[n_train:, target].to(torch.float64)
            pred_local = _ridge_predict(local_train, target_train, local_test, ridge=1e-3)
            pred_partner = _ridge_predict(partner_train, target_train, partner_test, ridge=1e-3)
            mse_local = torch.mean((pred_local - target_test) ** 2)
            mse_partner = torch.mean((pred_partner - target_test) ** 2)
            gains.append(float((mse_local - mse_partner).item()))

    block_size = 128 if samples == 4096 else max(1, samples // 32)
    block_count = samples // block_size
    return WorldAuditReport(
        analytic_latent_marginal_invariance=invariant,
        max_abs_mean_shift=mean_shift,
        max_abs_cov_shift=cov_shift,
        analytic_cross_lag_change=analytic_cross,
        empirical_cross_lag_change=empirical_cross,
        analytic_self_lag_norm=analytic_self,
        analytic_partner_lag_norm=analytic_partner,
        empirical_self_lag_change=sum(self_changes) / len(self_changes),
        empirical_partner_lag_change=sum(partner_changes) / len(partner_changes),
        min_partner_prediction_gain=min(gains),
        block_count=block_count,
        block_size=block_size,
        samples_per_regime=samples,
    )
