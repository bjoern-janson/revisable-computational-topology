from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from typing import Any

import torch

from .types import HandleState, NodeId, Operation


def clone_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {clone_tree(k): clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(x) for x in obj)
    return copy.deepcopy(obj)


@dataclass(frozen=True, order=True)
class EdgeIdentity:
    source: NodeId
    target: NodeId
    generation: int


@dataclass(frozen=True)
class EdgeSnapshot:
    identity: EdgeIdentity
    active_rank: int
    u: torch.Tensor
    v: torch.Tensor
    optimizer_state: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class ReopenHandle:
    handle_id: str
    identity: EdgeIdentity
    dormancy_version: int
    snapshot: EdgeSnapshot
    historical_support: float | None
    reopening_liability: float
    state: HandleState


@dataclass(frozen=True)
class StructuralTransaction:
    transaction_id: str
    proposal_id: str
    operation: Operation
    graph_before: str
    graph_after: str
    evidence_ref: str | None
    affected_routes: tuple[str, ...]
    reopen_handle_id: str | None
    charged_cost: float


class StructuralLedger:
    def __init__(self):
        self._transactions: list[StructuralTransaction] = []
        self._proposal_ids: set[str] = set()
        self._observations: list[dict[str, Any]] = []

    @property
    def transactions(self) -> tuple[StructuralTransaction, ...]:
        return tuple(self._transactions)

    @property
    def observations(self) -> tuple[dict[str, Any], ...]:
        return tuple(clone_tree(self._observations))

    def append_transaction(self, transaction: StructuralTransaction) -> None:
        if transaction.transaction_id in {t.transaction_id for t in self._transactions}:
            raise ValueError(f"duplicate transaction {transaction.transaction_id}")
        if transaction.proposal_id in self._proposal_ids:
            raise ValueError(f"proposal already applied {transaction.proposal_id}")
        self._transactions.append(transaction)
        self._proposal_ids.add(transaction.proposal_id)

    def append_observation(self, observation: dict[str, Any]) -> None:
        self._observations.append(clone_tree(observation))

    def proposal_applied(self, proposal_id: str) -> bool:
        return proposal_id in self._proposal_ids


def snapshot_digest(identity: EdgeIdentity, active_rank: int, u: torch.Tensor, v: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(f"{identity.source}|{identity.target}|{identity.generation}|{active_rank}".encode())
    h.update(u.detach().cpu().contiguous().numpy().tobytes())
    h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clone_snapshot(s: EdgeSnapshot) -> EdgeSnapshot:
    return EdgeSnapshot(
        identity=s.identity,
        active_rank=s.active_rank,
        u=s.u.clone(),
        v=s.v.clone(),
        optimizer_state=clone_tree(s.optimizer_state),
        digest=s.digest,
    )


def clone_handle(h: ReopenHandle) -> ReopenHandle:
    return ReopenHandle(
        handle_id=h.handle_id,
        identity=h.identity,
        dormancy_version=h.dormancy_version,
        snapshot=clone_snapshot(h.snapshot),
        historical_support=h.historical_support,
        reopening_liability=h.reopening_liability,
        state=h.state,
    )
