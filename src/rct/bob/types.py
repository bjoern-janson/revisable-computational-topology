from dataclasses import dataclass
from enum import Enum
from typing import Literal

NodeId = Literal["A", "B", "C", "D"]
NODE_IDS: tuple[NodeId, ...] = ("A", "B", "C", "D")


class Operation(str, Enum):
    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DORMANT = "DORMANT"
    RETIRE = "RETIRE"
    REOPEN = "REOPEN"


class ActionKind(str, Enum):
    PROBE = "PROBE"
    CREATE = "CREATE"
    MODIFY_UP = "MODIFY_UP"
    MODIFY_DOWN = "MODIFY_DOWN"
    DORMANT = "DORMANT"
    RETIRE = "RETIRE"
    REOPEN = "REOPEN"
    ABSTAIN = "ABSTAIN"


class EdgeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    RETIRED = "RETIRED"


class HandleState(str, Enum):
    VALID = "VALID"
    CONSUMED = "CONSUMED"
    REVOKED = "REVOKED"


@dataclass(frozen=True)
class ReopenLiability:
    info: float = 0.0
    reach: float = 0.0
    effective_use: float = 0.0

    @property
    def total(self) -> float:
        return self.info + self.reach + self.effective_use


@dataclass(frozen=True)
class CandidateRelation:
    source: NodeId
    target: NodeId


@dataclass(frozen=True)
class ProbeAcquisitionWarrant:
    warrant_id: str
    source: NodeId
    target: NodeId
    created_step: int
    expires_step: int


@dataclass(frozen=True)
class ProbeEvidence:
    evidence_id: str
    source: NodeId
    target: NodeId
    gain: float
    baseline_mse: float
    probed_mse: float
    cost: float
    created_step: int
    expires_step: int
    warrant_ref: str | None = None


@dataclass(frozen=True)
class ExistingEdgeEvidence:
    evidence_id: str
    source: NodeId
    target: NodeId
    generation: int
    revision_pressure: float
    contribution_ema: float
    usage_ema: float
    active_rank: int
    created_step: int
    expires_step: int


@dataclass(frozen=True)
class ReopenEvidence:
    evidence_id: str
    source: NodeId
    target: NodeId
    generation: int
    dormancy_version: int
    handle_id: str
    current_pressure: float
    historical_support: float | None
    reopening_liability: float
    created_step: int
    expires_step: int


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    kind: ActionKind
    source: NodeId | None = None
    target: NodeId | None = None
    generation: int | None = None
    dormancy_version: int | None = None
    handle_id: str | None = None
    evidence_ref: str | None = None
    rank_delta: int = 0


@dataclass(frozen=True)
class EditSpec:
    source: NodeId | None = None
    target: NodeId | None = None
    generation: int | None = None
    dormancy_version: int | None = None
    handle_id: str | None = None
    rank_delta: int = 0


@dataclass(frozen=True)
class StructuralProposal:
    proposal_id: str
    operation: Operation
    edit: EditSpec
    evidence_ref: str | None
    estimated_benefit: float = 0.0
    estimated_cost: float = 0.0
    affected_routes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    proposal_id: str
    allowed: bool
    reason: str
    charged_cost: float = 0.0


@dataclass(frozen=True)
class ExecutionAuthorization:
    design_commit: str
    implementation_commit: str
    authorization_id: str


@dataclass(frozen=True)
class BobStatus:
    design_commit: str
    implementation_state: str
    execution_state: str
    scientific_result: str
