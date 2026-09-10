from __future__ import annotations

import math
import torch

from .config import WorldConfig
from .types import NODE_IDS, NodeId


_PAIRINGS = {
    "W1": ((0, 1), (2, 3)),
    "W2": ((0, 1), (2, 3)),
    "W3": ((0, 2), (1, 3)),
}


class RotatingDependencyWorld:
    """Private rotating-dependency world.

    Learner-facing methods expose observations only. Audit helpers are explicitly
    named and must not be imported by manager-facing modules.
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self._world_gen = torch.Generator(device="cpu").manual_seed(config.world_seed)
        self._emission_gen = torch.Generator(device="cpu").manual_seed(config.emission_seed)
        self.phase_lengths = tuple(
            int(torch.randint(lo, hi + 1, (1,), generator=self._world_gen).item())
            for lo, hi in (
                (config.w1_steps_min, config.w1_steps_max),
                (config.w2_steps_min, config.w2_steps_max),
                (config.w3_steps_min, config.w3_steps_max),
            )
        )
        self.t = 0
        self._q = self._make_orthogonal(config.latent_dim)
        self._transition = {name: self._build_transition(name) for name in _PAIRINGS}
        self._z = torch.randn(4 * config.latent_dim, generator=self._world_gen, dtype=torch.float64)
        self._emissions = self._make_emissions()
        self._current = self._emit(self._z, self._emission_gen)

    @property
    def total_steps(self) -> int:
        return sum(self.phase_lengths)

    def _make_orthogonal(self, d: int) -> torch.Tensor:
        raw = torch.randn((d, d), generator=self._world_gen, dtype=torch.float64)
        q, r = torch.linalg.qr(raw)
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1.0
        return q * signs

    def _make_emissions(self):
        c = self.config
        params = {}
        for node in NODE_IDS:
            w1 = torch.randn((c.emission_hidden_dim, c.latent_dim), generator=self._emission_gen, dtype=torch.float64)
            w1 /= math.sqrt(c.latent_dim)
            w2 = torch.randn((c.obs_dim, c.emission_hidden_dim), generator=self._emission_gen, dtype=torch.float64)
            w2 /= math.sqrt(c.emission_hidden_dim)
            b1 = torch.zeros(c.emission_hidden_dim, dtype=torch.float64)
            b2 = torch.zeros(c.obs_dim, dtype=torch.float64)
            params[node] = (w1, w2, b1, b2)
        return params

    def _build_transition(self, regime: str) -> torch.Tensor:
        if regime not in _PAIRINGS:
            raise ValueError(f"unknown regime {regime!r}")
        c = self.config
        d = c.latent_dim
        r = torch.zeros((4 * d, 4 * d), dtype=torch.float64)
        q = self._q
        for i, j in _PAIRINGS[regime]:
            si = slice(i * d, (i + 1) * d)
            sj = slice(j * d, (j + 1) * d)
            r[si, si] = c.alpha * q
            r[si, sj] = c.beta * q
            r[sj, si] = -c.beta * q
            r[sj, sj] = c.alpha * q
        return r

    def _regime_at(self, t: int) -> str:
        if t < self.phase_lengths[0]:
            return "W1"
        if t < self.phase_lengths[0] + self.phase_lengths[1]:
            return "W2"
        return "W3"

    def _emit(self, z: torch.Tensor, generator: torch.Generator) -> dict[NodeId, torch.Tensor]:
        c = self.config
        out: dict[NodeId, torch.Tensor] = {}
        for idx, node in enumerate(NODE_IDS):
            zi = z[idx * c.latent_dim : (idx + 1) * c.latent_dim]
            w1, w2, b1, b2 = self._emissions[node]
            y = torch.tanh(w2 @ torch.tanh(w1 @ zi + b1) + b2)
            if c.emission_noise_std:
                y = y + c.emission_noise_std * torch.randn(y.shape, generator=generator, dtype=y.dtype)
            out[node] = y.to(torch.float32)
        return out

    def current_observation(self) -> dict[NodeId, torch.Tensor]:
        return {k: v.clone() for k, v in self._current.items()}

    def advance(self) -> dict[NodeId, torch.Tensor]:
        regime = self._regime_at(self.t)
        eps = torch.randn(self._z.shape, generator=self._world_gen, dtype=torch.float64)
        self._z = (
            self.config.rho * (self._transition[regime] @ self._z)
            + math.sqrt(1.0 - self.config.rho**2) * eps
        )
        self.t += 1
        self._current = self._emit(self._z, self._emission_gen)
        return self.current_observation()

    def audit_transition_matrix(self, regime: str) -> torch.Tensor:
        return self._transition[regime].clone()

    def audit_pairing(self, regime: str) -> tuple[tuple[NodeId, NodeId], ...]:
        return tuple((NODE_IDS[i], NODE_IDS[j]) for i, j in _PAIRINGS[regime])

    def audit_forced_sequence(
        self, regime: str, transitions: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transitions <= 0:
            raise ValueError("transitions must be positive")
        c = self.config
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise_gen = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
        z = torch.randn(4 * c.latent_dim, generator=gen, dtype=torch.float64)
        current_rows = []
        next_rows = []
        current = self._emit(z, noise_gen)
        r = self._transition[regime]
        for _ in range(transitions):
            eps = torch.randn(z.shape, generator=gen, dtype=torch.float64)
            z_next = c.rho * (r @ z) + math.sqrt(1.0 - c.rho**2) * eps
            nxt = self._emit(z_next, noise_gen)
            current_rows.append(torch.stack([current[n] for n in NODE_IDS]))
            next_rows.append(torch.stack([nxt[n] for n in NODE_IDS]))
            z, current = z_next, nxt
        return torch.stack(current_rows), torch.stack(next_rows)
