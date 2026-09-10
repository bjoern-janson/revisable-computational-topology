from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math

import torch
from torch import nn

from .lineage import (
    EdgeIdentity,
    EdgeSnapshot,
    ReopenHandle,
    clone_handle,
    clone_tree,
    snapshot_digest,
)
from .types import EdgeStatus, HandleState, NODE_IDS, NodeId


class InvalidHandle(RuntimeError):
    pass


class InvalidEdge(RuntimeError):
    pass


class LowRankInterface(nn.Module):
    def __init__(self, hidden_dim: int, max_rank: int, initial_rank: int):
        super().__init__()
        if not 1 <= initial_rank <= max_rank:
            raise ValueError("initial rank out of bounds")
        self.hidden_dim = hidden_dim
        self.max_rank = max_rank
        self.active_rank = initial_rank
        scale = 1.0 / math.sqrt(hidden_dim)
        self.u_factors = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * scale) for _ in range(max_rank)
        ])
        self.v_factors = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * scale) for _ in range(max_rank)
        ])

    @property
    def u(self) -> torch.Tensor:
        return torch.stack(list(self.u_factors), dim=1)

    @property
    def v(self) -> torch.Tensor:
        return torch.stack(list(self.v_factors), dim=0)

    def effective_weight(self) -> torch.Tensor:
        out = torch.zeros((self.hidden_dim, self.hidden_dim), dtype=self.u_factors[0].dtype)
        for k in range(self.active_rank):
            out = out + torch.outer(self.u_factors[k], self.v_factors[k])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for k in range(self.active_rank):
            out = out + self.u_factors[k] * torch.dot(self.v_factors[k], x)
        return out

    def load_factor_matrices(self, u: torch.Tensor, v: torch.Tensor, active_rank: int) -> None:
        if u.shape != (self.hidden_dim, self.max_rank) or v.shape != (self.max_rank, self.hidden_dim):
            raise ValueError("factor shape mismatch")
        with torch.no_grad():
            for k in range(self.max_rank):
                self.u_factors[k].copy_(u[:, k])
                self.v_factors[k].copy_(v[k, :])
        self.active_rank = active_rank


@dataclass
class EdgeRecord:
    identity: EdgeIdentity
    module: LowRankInterface
    optimizer: torch.optim.Optimizer | None
    status: EdgeStatus = EdgeStatus.ACTIVE
    age: int = 0
    usage_ema: float = 0.0
    contribution_ema: float = 0.0
    latest_positive_contribution: float | None = None
    dormancy_version: int = 0

    @property
    def active_rank(self) -> int:
        return self.module.active_rank


_USE_LATEST = object()


class AdapterGraph(nn.Module):
    def __init__(self, hidden_dim: int, max_rank: int, initial_rank: int, edge_lr: float = 1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_rank = max_rank
        self.initial_rank = initial_rank
        self.edge_lr = edge_lr
        self.modules_by_key = nn.ModuleDict()
        self._edges: dict[EdgeIdentity, EdgeRecord] = {}
        self._generation_counter: dict[tuple[NodeId, NodeId], int] = {}
        self._handles: dict[str, ReopenHandle] = {}
        self._handle_counter = 0

    @staticmethod
    def _key(identity: EdgeIdentity) -> str:
        return f"{identity.source}__{identity.target}__g{identity.generation}"

    def _new_optimizer(self, module: LowRankInterface) -> torch.optim.Optimizer:
        return torch.optim.AdamW(module.parameters(), lr=self.edge_lr)

    def create(self, source: NodeId, target: NodeId) -> EdgeRecord:
        if source == target or source not in NODE_IDS or target not in NODE_IDS:
            raise InvalidEdge("illegal endpoints")
        if self.has_active_endpoint(source, target):
            raise InvalidEdge("active endpoint pair already exists")
        pair = (source, target)
        generation = self._generation_counter.get(pair, 0) + 1
        self._generation_counter[pair] = generation
        identity = EdgeIdentity(source, target, generation)
        module = LowRankInterface(self.hidden_dim, self.max_rank, self.initial_rank)
        self.modules_by_key[self._key(identity)] = module
        edge = EdgeRecord(identity=identity, module=module, optimizer=self._new_optimizer(module))
        self._edges[identity] = edge
        return edge

    def edge(self, identity: EdgeIdentity) -> EdgeRecord:
        try:
            return self._edges[identity]
        except KeyError as exc:
            raise InvalidEdge(str(identity)) from exc

    def edges(self) -> tuple[EdgeRecord, ...]:
        return tuple(self._edges[k] for k in sorted(self._edges))

    def active_edges(self) -> tuple[EdgeRecord, ...]:
        return tuple(e for e in self.edges() if e.status is EdgeStatus.ACTIVE)

    def has_active_endpoint(self, source: NodeId, target: NodeId) -> bool:
        return any(
            e.identity.source == source and e.identity.target == target and e.status is EdgeStatus.ACTIVE
            for e in self._edges.values()
        )

    def is_active(self, identity: EdgeIdentity) -> bool:
        return self.edge(identity).status is EdgeStatus.ACTIVE

    def observe_contribution(self, identity: EdgeIdentity, c: float) -> None:
        edge = self.edge(identity)
        edge.contribution_ema = float(c)
        if c > 0.0:
            edge.latest_positive_contribution = float(c)

    def update_usage(self, identity: EdgeIdentity, usage: float) -> None:
        self.edge(identity).usage_ema = float(usage)

    def _snapshot(self, edge: EdgeRecord) -> EdgeSnapshot:
        if edge.optimizer is None:
            raise InvalidEdge("edge has no live optimizer")
        u, v = edge.module.u.detach().cpu(), edge.module.v.detach().cpu()
        return EdgeSnapshot(
            identity=edge.identity,
            active_rank=edge.active_rank,
            u=u.clone(),
            v=v.clone(),
            optimizer_state=clone_tree(edge.optimizer.state_dict()),
            digest=snapshot_digest(edge.identity, edge.active_rank, u, v),
        )

    def dormant(
        self,
        identity: EdgeIdentity,
        historical_support: float | None | object = _USE_LATEST,
        reopening_liability: float = 0.0,
    ) -> ReopenHandle:
        edge = self.edge(identity)
        if edge.status is not EdgeStatus.ACTIVE:
            raise InvalidEdge("only active edge may become dormant")
        support = edge.latest_positive_contribution if historical_support is _USE_LATEST else historical_support
        edge.dormancy_version += 1
        snapshot = self._snapshot(edge)
        self._handle_counter += 1
        handle_id = f"H{self._handle_counter:06d}"
        handle = ReopenHandle(
            handle_id=handle_id,
            identity=edge.identity,
            dormancy_version=edge.dormancy_version,
            snapshot=snapshot,
            historical_support=None if support is None else float(support),
            reopening_liability=float(reopening_liability),
            state=HandleState.VALID,
        )
        self._handles[handle_id] = handle
        edge.status = EdgeStatus.DORMANT
        return clone_handle(handle)

    def handle(self, handle_id: str) -> ReopenHandle:
        try:
            return clone_handle(self._handles[handle_id])
        except KeyError as exc:
            raise InvalidHandle(handle_id) from exc

    def handles(self) -> tuple[ReopenHandle, ...]:
        return tuple(self.handle(k) for k in sorted(self._handles))

    def handles_for_generation(self, identity: EdgeIdentity) -> tuple[ReopenHandle, ...]:
        return tuple(h for h in self.handles() if h.identity == identity)

    def reopen(self, handle_id: str) -> EdgeRecord:
        if handle_id not in self._handles:
            raise InvalidHandle(handle_id)
        internal = self._handles[handle_id]
        if internal.state is not HandleState.VALID:
            raise InvalidHandle(f"handle {handle_id} is {internal.state.value}")
        edge = self.edge(internal.identity)
        if edge.status is not EdgeStatus.DORMANT:
            raise InvalidHandle("generation is not dormant")
        if internal.dormancy_version != edge.dormancy_version:
            raise InvalidHandle("stale dormancy version")
        if any(
            other.identity != edge.identity
            and other.identity.source == edge.identity.source
            and other.identity.target == edge.identity.target
            and other.status is EdgeStatus.ACTIVE
            for other in self._edges.values()
        ):
            raise InvalidHandle("endpoint conflict")
        if edge.optimizer is None:
            edge.optimizer = self._new_optimizer(edge.module)
        edge.module.load_factor_matrices(
            internal.snapshot.u,
            internal.snapshot.v,
            internal.snapshot.active_rank,
        )
        edge.optimizer.load_state_dict(clone_tree(internal.snapshot.optimizer_state))
        edge.status = EdgeStatus.ACTIVE
        edge.latest_positive_contribution = None
        self._handles[handle_id] = replace(internal, state=HandleState.CONSUMED)
        return edge

    def retire(self, identity: EdgeIdentity) -> None:
        edge = self.edge(identity)
        if edge.status is not EdgeStatus.ACTIVE:
            raise InvalidEdge("only active edge may retire")
        edge.status = EdgeStatus.RETIRED
        edge.optimizer = None
        for handle_id, h in list(self._handles.items()):
            if h.identity == identity and h.state is HandleState.VALID:
                self._handles[handle_id] = replace(h, state=HandleState.REVOKED)

    def modify_rank(self, identity: EdgeIdentity, delta: int) -> None:
        if delta not in (-1, 1):
            raise ValueError("MODIFY changes rank by exactly one")
        edge = self.edge(identity)
        if edge.status is not EdgeStatus.ACTIVE:
            raise InvalidEdge("only active edge may be modified")
        new_rank = edge.active_rank + delta
        if not 1 <= new_rank <= self.max_rank:
            raise InvalidEdge("rank bound")
        edge.module.active_rank = new_rank

    def route(self, latents: dict[NodeId, torch.Tensor]) -> dict[NodeId, list[torch.Tensor]]:
        out: dict[NodeId, list[torch.Tensor]] = {n: [] for n in NODE_IDS}
        for edge in self.active_edges():
            out[edge.identity.target].append(edge.module(latents[edge.identity.source]))
        return out

    def edge_optimizer_step(self, identity: EdgeIdentity, loss: torch.Tensor) -> None:
        edge = self.edge(identity)
        if edge.status is not EdgeStatus.ACTIVE or edge.optimizer is None:
            raise InvalidEdge("edge is not optimizable")
        edge.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        edge.optimizer.step()

    def synthetic_edge_step(self, identity: EdgeIdentity) -> None:
        edge = self.edge(identity)
        x = torch.linspace(-1.0, 1.0, self.hidden_dim)
        y = edge.module(x)
        loss = (y.square().mean() + 0.1 * y.sum())
        self.edge_optimizer_step(identity, loss)

    def optimizer_fingerprint(self, identity: EdgeIdentity, inactive_from: int = 0) -> str:
        edge = self.edge(identity)
        h = hashlib.sha256()
        if edge.optimizer is None:
            h.update(b"NONE")
            return h.hexdigest()
        for idx, p in enumerate(edge.module.parameters()):
            if idx < self.max_rank:
                k = idx
            else:
                k = idx - self.max_rank
            if k < inactive_from:
                continue
            state = edge.optimizer.state.get(p, {})
            h.update(str(k).encode())
            for name in sorted(state):
                val = state[name]
                h.update(name.encode())
                if torch.is_tensor(val):
                    h.update(val.detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(repr(val).encode())
        return h.hexdigest()

    def parameter_fingerprint(self) -> str:
        h = hashlib.sha256()
        for edge in self.edges():
            h.update(repr(edge.identity).encode())
            h.update(edge.status.value.encode())
            h.update(edge.module.u.detach().cpu().contiguous().numpy().tobytes())
            h.update(edge.module.v.detach().cpu().contiguous().numpy().tobytes())
            h.update(str(edge.active_rank).encode())
            if edge.optimizer is not None:
                h.update(repr(clone_tree(edge.optimizer.state_dict())).encode())
        return h.hexdigest()

    def structural_fingerprint(self) -> str:
        h = hashlib.sha256()
        for edge in self.edges():
            h.update(
                f"{edge.identity.source}|{edge.identity.target}|{edge.identity.generation}|{edge.status.value}|{edge.active_rank}".encode()
            )
        for handle_id in sorted(self._handles):
            handle = self._handles[handle_id]
            h.update(f"{handle_id}|{handle.dormancy_version}|{handle.state.value}".encode())
        return h.hexdigest()
