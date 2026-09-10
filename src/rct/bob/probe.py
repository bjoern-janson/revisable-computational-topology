from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import copy

import torch

from .types import CandidateRelation, NodeId, ProbeEvidence


class ProbeBudgetExhausted(RuntimeError):
    pass


class EvidenceError(RuntimeError):
    pass


class ProbeBudget:
    def __init__(self, max_probes: int = 2, cost_per_probe: float = 0.02):
        self.max_probes = int(max_probes)
        self.cost_per_probe = float(cost_per_probe)
        self.remaining = self.max_probes
        self.spent_window = 0.0
        self.spent_lifetime = 0.0

    def charge(self) -> float:
        if self.remaining <= 0:
            raise ProbeBudgetExhausted("probe budget exhausted")
        self.remaining -= 1
        self.spent_window += self.cost_per_probe
        self.spent_lifetime += self.cost_per_probe
        return self.cost_per_probe

    def reset_window(self) -> None:
        self.remaining = self.max_probes
        self.spent_window = 0.0


@dataclass(frozen=True)
class BufferRow:
    step: int
    latents: dict[NodeId, torch.Tensor]
    residuals: dict[NodeId, torch.Tensor]
    encoder_fingerprint: str


class LatentResidualBuffer:
    def __init__(self, maxlen: int = 256):
        self.maxlen = maxlen
        self._rows: deque[BufferRow] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        step: int,
        latents: dict[NodeId, torch.Tensor],
        residuals: dict[NodeId, torch.Tensor],
        *,
        encoder_fingerprint: str,
        target_revealed: bool,
    ) -> None:
        if not target_revealed:
            raise ValueError("probe rows require already revealed targets")
        self._rows.append(BufferRow(
            step=int(step),
            latents={k: v.detach().cpu().clone() for k, v in latents.items()},
            residuals={k: v.detach().cpu().clone() for k, v in residuals.items()},
            encoder_fingerprint=str(encoder_fingerprint),
        ))

    def last(self, n: int) -> tuple[BufferRow, ...]:
        if n > len(self._rows):
            raise ValueError(f"need {n} rows, have {len(self._rows)}")
        rows = list(self._rows)[-n:]
        return tuple(BufferRow(
            r.step,
            {k: v.clone() for k, v in r.latents.items()},
            {k: v.clone() for k, v in r.residuals.items()},
            r.encoder_fingerprint,
        ) for r in rows)


@dataclass
class _EvidenceState:
    evidence: Any
    consumed: bool = False


class EvidenceRegistry:
    def __init__(self):
        self._entries: dict[str, _EvidenceState] = {}
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"E{self._counter:06d}"

    def add(self, evidence: Any) -> None:
        evidence_id = evidence.evidence_id
        if evidence_id in self._entries:
            raise EvidenceError(f"duplicate evidence {evidence_id}")
        self._entries[evidence_id] = _EvidenceState(copy.deepcopy(evidence), False)
        if evidence_id.startswith("E") and evidence_id[1:].isdigit():
            self._counter = max(self._counter, int(evidence_id[1:]))

    def get(self, evidence_id: str) -> Any:
        if evidence_id not in self._entries:
            raise EvidenceError(f"unknown evidence {evidence_id}")
        return copy.deepcopy(self._entries[evidence_id].evidence)

    def consumed(self, evidence_id: str) -> bool:
        if evidence_id not in self._entries:
            raise EvidenceError(f"unknown evidence {evidence_id}")
        return self._entries[evidence_id].consumed

    def is_usable(self, evidence_id: str, current_step: int) -> bool:
        if evidence_id not in self._entries:
            return False
        entry = self._entries[evidence_id]
        if entry.consumed:
            return False
        ev = entry.evidence
        return int(ev.created_step) <= int(current_step) <= int(ev.expires_step)

    def consume(self, evidence_id: str) -> None:
        if evidence_id not in self._entries:
            raise EvidenceError(f"unknown evidence {evidence_id}")
        entry = self._entries[evidence_id]
        if entry.consumed:
            raise EvidenceError(f"already consumed {evidence_id}")
        entry.consumed = True

    def items(self) -> tuple[Any, ...]:
        return tuple(copy.deepcopy(x.evidence) for _, x in sorted(self._entries.items()))


def relation_probe(
    relation: CandidateRelation,
    buffer: LatentResidualBuffer,
    budget: ProbeBudget,
    registry: EvidenceRegistry,
    *,
    current_step: int,
    ridge_lambda: float = 1e-3,
    window: int = 256,
) -> ProbeEvidence:
    rows = buffer.last(window)
    split = window // 2
    x = torch.stack([r.latents[relation.source] for r in rows]).to(torch.float64)
    y = torch.stack([r.residuals[relation.target] for r in rows]).to(torch.float64)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    eye = torch.eye(x_train.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(x_train.T @ x_train + ridge_lambda * eye, x_train.T @ y_train)
    pred = x_test @ w
    baseline_mse = float(torch.mean(y_test.square()).item())
    probed_mse = float(torch.mean((y_test - pred).square()).item())
    gain = baseline_mse - probed_mse
    cost = budget.charge()
    ev = ProbeEvidence(
        evidence_id=registry.next_id(),
        source=relation.source,
        target=relation.target,
        gain=gain,
        baseline_mse=baseline_mse,
        probed_mse=probed_mse,
        cost=cost,
        created_step=int(current_step),
        expires_step=int(current_step) + 64,
    )
    registry.add(ev)
    return ev
