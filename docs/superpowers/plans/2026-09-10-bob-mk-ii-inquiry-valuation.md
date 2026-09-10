# Bob Mk II — Learned Inquiry and Structural Action Valuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Bob Mk II architecture as one continuous neural organism whose fixed-topology backbone and mutable adapter graph are inherited from repaired Bob Mk I, while a new inspectable linear valuation learns from completed 256-step realized returns to rank PROBE, CREATE, MODIFY, DORMANT, RETIRE, REOPEN, and ABSTAIN without bypassing typed evidence or the authoritative structural gate.

**Architecture:** Python/PyTorch package `rct.bob`. The repaired Mk I world, neural backbone, directed low-rank interfaces, typed evidence, structural lineage, gate, and fail-closed execution boundary are implemented first. Mk II then adds a 38-dimensional hand-auditable action feature vector, metadata-only future-access dry runs, a persistent completed-return ridge regressor `eta`, block-balanced score-independent exploration for non-RETIRE actions, and a separate retirement-support gate. Candidate generation remains bounded to the fixed 12 directed non-self endpoint pairs and is intentionally non-learning in this implementation so that `eta` is the only new manager-learning channel.

**Tech Stack:** Python 3.11+, PyTorch 2.2+, NumPy 1.26+, Matplotlib 3.8+, pytest 8+; standard-library `dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, `argparse`, `ast`, and `collections`.

**Spec:** `docs/superpowers/specs/2026-09-10-bob-mk-ii-inquiry-valuation-design.md` at approved design commit `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`. Inherited Mk I authority is `docs/superpowers/specs/2026-09-10-bob-mk-i-repair-freeze.md` at `deb0fd205b395684531720cdd0a2cd16b9f10ae0` plus the terminal Mk I plan amendment at `37f8b3df554ddffcfdf4b17a51b3fd703294ae78`.

## Global Constraints

- One continuing lifetime: no reset of graph, structural lineage, topology-manager state, `eta`, exposure counters, or delayed-return ledger across W1/W2/W3.
- Preserve separate observability of `theta` plasticity, interface-parameter `phi` plasticity, structural `G` plasticity, `eta` plasticity, and REOPEN events.
- Endpoint vocabulary is exactly `A,B,C,D`; candidate relation generation is bounded to the 12 directed non-self endpoint pairs.
- Structural vocabulary is exactly `CREATE`, `MODIFY`, `DORMANT`, `RETIRE`, `REOPEN`; `MERGE` and `SPLIT` have no constructor, gate path, or application path.
- Mk II choice vocabulary is exactly `PROBE`, `CREATE`, `MODIFY_UP`, `MODIFY_DOWN`, `DORMANT`, `RETIRE`, `REOPEN`, `ABSTAIN`.
- Evidence, candidate, valuation, proposal, permission, and mutation remain different typed stages.
- The manager never receives regime ID, hidden transition operator/pairing, correct-edge labels, audit-private pair scores, privileged future loss, future probe answers, or alternate world rollouts.
- `RELATION_PROBE` remains temporary and read-only with respect to persistent graph/backbone, uses frozen ridge `1e-3`, and is limited to 2 probes per 64-step structural window with lifetime spend monotone.
- Mk I loss units remain coordinate-mean MSE per target and **sum over the four targets** for `L_full` and `L_local`.
- World construction remains `alpha=0.6`, `beta=0.8`, `rho=0.85`, stationary isotropic latent dynamics, W1/W2 pairing AB/CD, W3 pairing AC/BD, and private phase lengths sampled independently from inclusive `[1536,2560]` using the world RNG.
- `CREATE` starts a fresh edge generation. `DORMANT` creates a dormancy-versioned single-use handle with matching parameter+optimizer snapshot. Successful REOPEN consumes that handle. `RETIRE` revokes all still-valid handles for the generation.
- Historical reopen support is the latest same-active-epoch contribution satisfying exactly `c(E) > 0`; zero and negative values do not qualify and the `0.01` consequentiality threshold is not reused here.
- `C_retained` and `C_reachable` are route-level descriptive telemetry. They may be input features or feature deltas but are never direct reward terms.
- Manager training horizon is exactly `H=256` ordinary steps, `gamma=1`, with target `Y_t = -sum(L_task + probe_cost + edit_cost + maintenance_cost + reopen_cost)` over steps `t..t+255`.
- `Y_t` is an observed continuation return, not a causal action effect.
- `eta_0 = 0`, completed-return ridge only, no TD target, no Bellman backup, no target network.
- `eta_ridge_lambda = 1.0`, fit in deterministic CPU `torch.float64` using `torch.linalg.solve(X.T @ X + lambda*I, X.T @ y)`.
- Feature vector has exactly 38 coordinates in the order frozen in Task 7; there are no learned feature embeddings and no interaction terms in Mk II V0.
- Feature numeric epsilon is `1e-6`; signed normalized ratios are clipped to `[-2,2]`.
- Candidate relation generator emits at most 2 absent directed relation candidates per structural boundary, sampled uniformly without replacement from legal absent pairs using the independent manager RNG seed `31`. It does not inspect pair-specific absent-edge evidence.
- Exploration schedule uses independent seed `43` and exact block balance: in each consecutive block of 4 structural decision boundaries, exactly 1 slot is selected uniformly for score-independent exploration.
- Exploration selects only among eligible non-RETIRE actions and minimizes `(action_class_exposure_count, action_identity_exposure_count, stable_tie_hash)`.
- Stable greedy/exploration tie hash is SHA-256 of `"47|<canonical_candidate_id>"`; lower hex digest wins.
- RETIRE is never forced. Retirement support requires at least 6 matured non-RETIRE rows in the exact frozen coarse context region from Task 9 and at least 2 distinct non-RETIRE action kinds in those rows.
- No scientific W1→W2→W3 lifetime may run during implementation, tests, CI, validation, or plan execution. Only unit tests, environment audit, synthetic component tests, and fixed-W1 conformance are allowed.
- The eventual scientific runner must fail closed before world construction unless explicit authorization binds exact Mk II design commit and exact terminal implementation commit and the checked-out executable source is clean.

## File Structure

```text
pyproject.toml
README.md
.github/workflows/test.yml
src/rct/__init__.py
src/rct/bob/__init__.py
src/rct/bob/config.py
src/rct/bob/types.py
src/rct/bob/world.py
src/rct/bob/audit.py
src/rct/bob/backbone.py
src/rct/bob/interfaces.py
src/rct/bob/lineage.py
src/rct/bob/probe.py
src/rct/bob/gate.py
src/rct/bob/metrics.py
src/rct/bob/features.py
src/rct/bob/valuation.py
src/rct/bob/exploration.py
src/rct/bob/manager.py
src/rct/bob/lifetime.py
src/rct/bob/trajectory.py
src/rct/bob/visualize.py
src/rct/bob/authorization.py
src/rct/bob/cli.py
tests/test_types_and_config.py
tests/test_world.py
tests/test_world_audit.py
tests/test_backbone.py
tests/test_interfaces.py
tests/test_lineage.py
tests/test_probe.py
tests/test_gate.py
tests/test_metrics.py
tests/test_features.py
tests/test_valuation.py
tests/test_exploration.py
tests/test_manager.py
tests/test_lifetime.py
tests/test_trajectory.py
tests/test_authorization.py
tests/test_cli.py
tests/test_firewall.py
docs/status/BOB_MK_II_STATUS.md
```

---

### Task 1: Package skeleton, frozen types, and exact configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/rct/__init__.py`
- Create: `src/rct/bob/__init__.py`
- Create: `src/rct/bob/config.py`
- Create: `src/rct/bob/types.py`
- Test: `tests/test_types_and_config.py`

**Interfaces:**
- Produces: `NodeId`, `NODE_IDS`, `Operation`, `ActionKind`, `EdgeStatus`, `HandleState`, `ReopenLiability`, `CandidateRelation`, `ProbeEvidence`, `ExistingEdgeEvidence`, `ReopenEvidence`, `ActionCandidate`, `StructuralProposal`, `GateDecision`, `ExecutionAuthorization`, `BobConfig`, `WorldConfig`, `MkIIConfig`.

- [ ] **Step 1: Write RED type/config tests**

```python
from rct.bob.config import BobConfig, WorldConfig, MkIIConfig
from rct.bob.types import NODE_IDS, Operation, ActionKind


def test_exact_vocabularies_and_mkii_constants():
    assert NODE_IDS == ("A", "B", "C", "D")
    assert {x.value for x in Operation} == {"CREATE", "MODIFY", "DORMANT", "RETIRE", "REOPEN"}
    assert {x.value for x in ActionKind} == {
        "PROBE", "CREATE", "MODIFY_UP", "MODIFY_DOWN",
        "DORMANT", "RETIRE", "REOPEN", "ABSTAIN",
    }
    m = MkIIConfig()
    assert m.return_horizon_steps == 256
    assert m.eta_ridge_lambda == 1.0
    assert m.feature_dim == 38
    assert m.exploration_block_size == 4
    assert m.exploration_slots_per_block == 1
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_types_and_config.py -v`
Expected: FAIL because `rct.bob` does not exist.

- [ ] **Step 3: Add packaging**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "revisable-computational-topology"
version = "0.0.2"
requires-python = ">=3.11"
dependencies = ["torch>=2.2", "numpy>=1.26", "matplotlib>=3.8"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Implement exact enums and candidate/evidence records**

```python
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

class HandleState(str, Enum):
    VALID = "VALID"
    CONSUMED = "CONSUMED"
    REVOKED = "REVOKED"

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
```

`ProbeEvidence`, `ExistingEdgeEvidence`, and `ReopenEvidence` each include `created_step`, `expires_step`, exact typed identity fields, and immutable evidence IDs. `ReopenEvidence` additionally includes `historical_support: float | None` and authoritative `reopening_liability`.

- [ ] **Step 5: Implement exact config values**

```python
@dataclass(frozen=True)
class BobConfig:
    hidden_dim: int = 32
    interface_max_rank: int = 8
    interface_initial_rank: int = 2
    structural_interval: int = 64
    probe_window: int = 256
    probes_per_window: int = 2
    probe_cost: float = 0.02
    create_cost: float = 0.25
    modify_cost: float = 0.10
    dormant_cost: float = 0.05
    retire_cost: float = 0.05
    edit_budget_per_window: float = 0.50
    maintenance_cost_per_rank: float = 0.001
    manager_warmup_steps: int = 256
    local_loss_weight: float = 1.0
    task_opportunity_min_gain: float = 0.02

@dataclass(frozen=True)
class WorldConfig:
    latent_dim: int = 8
    obs_dim: int = 16
    emission_hidden_dim: int = 32
    rho: float = 0.85
    alpha: float = 0.6
    beta: float = 0.8
    emission_noise_std: float = 0.05
    w1_steps_min: int = 1536
    w1_steps_max: int = 2560
    w2_steps_min: int = 1536
    w2_steps_max: int = 2560
    w3_steps_min: int = 1536
    w3_steps_max: int = 2560
    world_seed: int = 7
    emission_seed: int = 17
    network_seed: int = 29
    manager_seed: int = 31
    probe_seed: int = 37
    audit_seed: int = 41

@dataclass(frozen=True)
class MkIIConfig:
    return_horizon_steps: int = 256
    eta_ridge_lambda: float = 1.0
    feature_dim: int = 38
    feature_eps: float = 1e-6
    feature_clip: float = 2.0
    relation_candidates_per_boundary: int = 2
    exploration_block_size: int = 4
    exploration_slots_per_block: int = 1
    exploration_seed: int = 43
    tie_break_seed: int = 47
    retire_support_min_rows: int = 6
    retire_support_min_action_kinds: int = 2
```

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_types_and_config.py -v`
Expected: PASS.

```bash
git add pyproject.toml src/rct tests/test_types_and_config.py
git commit -m "feat: define Bob Mk II package contracts"
```

---

### Task 2: Repaired rotating world and independent environment audit

**Files:**
- Create: `src/rct/bob/world.py`
- Create: `src/rct/bob/audit.py`
- Test: `tests/test_world.py`
- Test: `tests/test_world_audit.py`

**Interfaces:**
- Produces `RotatingDependencyWorld.current_observation()`, `advance()`, and audit-only forced-regime helpers.
- Produces `WorldAuditReport` with analytic invariance and finite ridge-opportunity fields.

- [ ] **Step 1: Write RED world tests**

```python
def test_private_schedule_is_sampled_and_not_exposed_in_observation():
    w = RotatingDependencyWorld(WorldConfig())
    assert 1536 <= w.phase_lengths[0] <= 2560
    assert set(w.current_observation()) == {"A", "B", "C", "D"}
    assert not hasattr(w.current_observation(), "regime")
```

- [ ] **Step 2: Implement exact self+partner orthogonal transition**

For every paired `(i,j)` block use:

```python
R = torch.cat([
    torch.cat([alpha * Q, beta * Q], dim=1),
    torch.cat([-beta * Q, alpha * Q], dim=1),
], dim=0)
z_next = rho * (R_regime @ z) + (1.0 - rho**2) ** 0.5 * eps
```

W1/W2 pair AB and CD; W3 pairs AC and BD. Initialize `z_0 ~ N(0,I)`. Emissions are fixed `tanh(W2 @ tanh(W1 @ z + b1) + b2)` with fan-in-normal weights, zero biases, fixed noise, and independent emission RNG.

- [ ] **Step 3: Implement finite environment opportunity audit**

Use exactly 4096 forced-regime transitions per audited regime. Fit ridge `lambda=1e-3` on the first 3072 and evaluate the final 1024. Compare target-local current observation against target-local + true-partner current observation. Return only the minimum held-out directed-pair gain outside the audit-private table. Aggregate serial summaries in 32 contiguous blocks of 128 transitions.

- [ ] **Step 4: Run GREEN and commit**

Run: `pytest tests/test_world.py tests/test_world_audit.py -v`
Expected: PASS.

```bash
git add src/rct/bob/world.py src/rct/bob/audit.py tests/test_world.py tests/test_world_audit.py
git commit -m "feat: add repaired rotating dependency world"
```

---

### Task 3: Fixed-topology neural backbone with separate local/full heads

**Files:**
- Create: `src/rct/bob/backbone.py`
- Test: `tests/test_backbone.py`

**Interfaces:**
- Produces `BobBackbone.encode`, `decode_local`, `decode_full`, `losses`.

- [ ] **Step 1: Write RED tests for separate heads and exact loss aggregation**

```python
def test_local_and_full_heads_are_distinct_and_losses_sum_domains():
    model = BobBackbone(WorldConfig(), BobConfig())
    assert model.local_decoders is not model.full_decoders
    losses = model.aggregate_losses({k: torch.tensor(float(i + 1)) for i, k in enumerate(NODE_IDS)})
    assert losses == 10.0
```

- [ ] **Step 2: Implement encoders and heads**

Each node encoder is `Linear(obs_dim,64) -> GELU -> Linear(64,hidden_dim)`. Local head is `Linear(hidden_dim,64) -> GELU -> Linear(64,obs_dim)`. Full head receives `[h_j, sum(incoming_messages)]` and is `Linear(2*hidden_dim,64) -> GELU -> Linear(64,obs_dim)`.

- [ ] **Step 3: Implement exact training losses**

Per target use coordinate-mean MSE. Sum four local losses and four full losses. Training loss is `L_full + local_loss_weight * L_local`.

- [ ] **Step 4: Run GREEN and commit**

Run: `pytest tests/test_backbone.py -v`
Expected: PASS.

```bash
git add src/rct/bob/backbone.py tests/test_backbone.py
git commit -m "feat: add Bob local and full prediction heads"
```

---

### Task 4: Directed interfaces, generation-safe lifecycle, and optimizer custody

**Files:**
- Create: `src/rct/bob/interfaces.py`
- Create: `src/rct/bob/lineage.py`
- Test: `tests/test_interfaces.py`
- Test: `tests/test_lineage.py`

**Interfaces:**
- Produces `LowRankInterface`, `AdapterGraph`, `EdgeSnapshot`, `ReopenHandle`, `StructuralTransaction`, `StructuralLedger`.

- [ ] **Step 1: Write RED lifecycle tests**

```python
def test_consumed_handle_cannot_reopen_after_new_dormancy(graph_fixture):
    g = graph_fixture
    e = g.create("A", "B")
    h1 = g.dormant(e.identity, historical_support=0.2)
    g.reopen(h1.handle_id)
    g.synthetic_edge_step(e.identity)
    h2 = g.dormant(e.identity, historical_support=0.1)
    before = g.parameter_fingerprint()
    with pytest.raises(InvalidHandle):
        g.reopen(h1.handle_id)
    assert g.parameter_fingerprint() == before
    g.reopen(h2.handle_id)
```

- [ ] **Step 2: Implement edge identity and low-rank parameters**

`EdgeIdentity = (source,target,generation)` with monotone generation per endpoint pair. Each edge stores `u[hidden_dim,max_rank]`, `v[max_rank,hidden_dim]`, active rank, status, age, usage/contribution summaries, and active-epoch `latest_positive_contribution: float | None`.

- [ ] **Step 3: Implement exact active-epoch support rule**

```python
def observe_contribution(edge, c: float) -> None:
    if c > 0.0:
        edge.latest_positive_contribution = float(c)
```

CREATE and successful REOPEN set `latest_positive_contribution=None`. DORMANT freezes its current value into the new handle. No other epoch's positive value carries forward.

- [ ] **Step 4: Implement snapshot/optimizer lifecycle atomically**

CREATE allocates fresh per-edge AdamW state. DORMANT snapshots exact edge parameters and that optimizer state, assigns monotone `dormancy_version`, marks edge inactive, and creates a VALID handle. REOPEN validates exact generation+version, restores the matching snapshot, marks handle CONSUMED, and starts a new active epoch. RETIRE destroys live optimizer state and marks every still-VALID generation handle REVOKED. MODIFY_DOWN freezes removed factors and their moments; MODIFY_UP resumes them.

- [ ] **Step 5: Test no dormant/inactive drift**

Take AdamW steps on other parameters after DORMANT and after rank-down; assert frozen edge/rank factor bytes and stored moments remain identical.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_interfaces.py tests/test_lineage.py -v`
Expected: PASS.

```bash
git add src/rct/bob/interfaces.py src/rct/bob/lineage.py tests/test_interfaces.py tests/test_lineage.py
git commit -m "feat: add generation safe structural lifecycle"
```

---

### Task 5: Pair-specific probe service and typed evidence registry

**Files:**
- Create: `src/rct/bob/probe.py`
- Modify: `src/rct/bob/lineage.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Produces `ProbeBudget`, `LatentResidualBuffer`, `EvidenceRegistry`, `relation_probe`.

- [ ] **Step 1: Write RED budget/non-persistence tests**

```python
def test_probe_budget_resets_window_not_lifetime(probe_fixture):
    budget = probe_fixture.budget
    probe_fixture.buy("A", "B")
    probe_fixture.buy("C", "D")
    with pytest.raises(ProbeBudgetExhausted):
        probe_fixture.buy("A", "C")
    assert budget.spent_lifetime == pytest.approx(0.04)
    budget.reset_window()
    assert budget.remaining == 2
    assert budget.spent_lifetime == pytest.approx(0.04)
```

- [ ] **Step 2: Implement restricted pair-query service**

The manager passes exactly one previously generated relation token. The service internally reads the last 256 detached, already-target-revealed latent/residual rows, fits chronological 128/128 ridge with `1e-3`, returns `ProbeEvidence`, destroys fitted `W`, and never exposes the all-domain buffer or scores for other pairs.

- [ ] **Step 3: Implement one-window evidence expiry/consumption**

Evidence created at boundary step `t` gets `expires_step=t+64`. CREATE at the next boundary may use it if `current_step <= expires_step`; accepted mutation consumes it. Rejected proposals do not mutate evidence, and stale evidence stays historical but cannot be reused.

- [ ] **Step 4: Run GREEN and commit**

Run: `pytest tests/test_probe.py -v`
Expected: PASS.

```bash
git add src/rct/bob/probe.py src/rct/bob/lineage.py tests/test_probe.py
git commit -m "feat: add typed relation evidence service"
```

---

### Task 6: Authoritative gate, eligibility preview, and transactional application

**Files:**
- Create: `src/rct/bob/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Produces `GateContext`, `EligibilityPreview`, `StructuralGate.preview`, `StructuralGate.decide`, `apply_accepted_proposal`.

- [ ] **Step 1: Write RED legality/not-truth tests**

```python
def test_high_value_cannot_bypass_missing_evidence(gate_fixture):
    c = ActionCandidate("C1", ActionKind.CREATE, source="A", target="B", evidence_ref=None)
    preview = gate_fixture.gate.preview(c, gate_fixture.ctx)
    assert not preview.formable


def test_gate_does_not_receive_hidden_world_fields():
    names = {f.name for f in dataclasses.fields(GateContext)}
    assert names.isdisjoint({"regime", "pairing", "transition_operator", "correct_edges", "future_loss"})
```

- [ ] **Step 2: Implement authoritative immutable views**

Gate context includes exact current edge generation/status/rank, handle state/version/liability, evidence type/freshness/consumed state, proposal state, edit/probe budgets, cooldown, and current pressure. It derives operation charges from config; `proposal.estimated_cost` is never authoritative.

- [ ] **Step 3: Implement preview and final decide with same legality core**

`preview(candidate, ctx)` is side-effect-free and may only answer whether the candidate is currently formable and what authoritative deterministic charges/bookkeeping would apply. `decide(proposal, fresh_ctx)` reruns the same predicates immediately before mutation. It rejects stale generations, invalid handles, wrong dormancy versions, stale/consumed evidence, rank bounds, duplicate endpoint generation, insufficient budget, cooldown, and duplicate proposal IDs.

- [ ] **Step 4: Implement exact-once transactional application**

Persist proposal -> persist gate decision -> apply graph edit once -> persist transaction with before/after structural fingerprints and consumed evidence/handle IDs. A second application of the same proposal must produce no charge, mutation, optimizer registration, or transaction.

- [ ] **Step 5: Run GREEN and commit**

Run: `pytest tests/test_gate.py tests/test_interfaces.py tests/test_lineage.py -v`
Expected: PASS.

```bash
git add src/rct/bob/gate.py tests/test_gate.py
git commit -m "feat: add authoritative Bob structural gate"
```

---

### Task 7: Route-level telemetry and exact 38-coordinate action feature schema

**Files:**
- Create: `src/rct/bob/metrics.py`
- Create: `src/rct/bob/features.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces `CorrectiveAccessProfile`, `CorrectionOutcome`, `FeatureContext`, `FutureDryRun`, `FEATURE_NAMES`, `action_features(candidate, ctx) -> torch.Tensor`.

- [ ] **Step 1: Implement route-level access profile first**

`CorrectionRoute` identity is `(commitment_id, route_id, operation, edge_generation, target_rank_or_handle)`. Persistent route cohort membership never disappears. `C_retained` counts route state/snapshot still present and not consumed/revoked. `C_reachable` additionally requires current lifecycle/budget/cooldown/gate admissibility. Empty cohort returns `None`, never `1.0`. Demonstrated correction is a separate ledger record.

- [ ] **Step 2: Freeze and test feature ordering exactly**

`FEATURE_NAMES` has exactly these 38 entries:

```text
00 bias
01 pressure_ratio
02 contribution_ratio
03 usage_ema
04 probe_gain_ratio
05 evidence_freshness
06 rank_fraction
07 probe_budget_fraction
08 edit_budget_fraction
09 probe_charge_fraction
10 edit_charge_fraction
11 maintenance_delta_fraction
12 reopen_liability_fraction
13 total_immediate_charge_fraction
14 route_profile_defined
15 delta_retained
16 delta_reachable
17 post_probe_headroom
18 post_formable_nonretire_class_fraction
19 delta_valid_handle_fraction
20 reachable_route_concentration_after
21 produces_evidence_flag
22 edge_age_fraction
23 generation_fraction
24 dormancy_version_fraction
25 historical_support_ratio
26 time_since_related_action_fraction
27 accepted_related_fraction
28 rejected_related_fraction
29 prior_correction_exercised
30 action_PROBE
31 action_CREATE
32 action_MODIFY_UP
33 action_MODIFY_DOWN
34 action_DORMANT
35 action_RETIRE
36 action_REOPEN
37 action_ABSTAIN
```

No interactions are appended.

- [ ] **Step 3: Implement exact normalization functions**

```python
def signed_ratio(x: float, scale: float, eps: float = 1e-6) -> float:
    return max(-2.0, min(2.0, x / max(abs(scale), eps)))
```

Use target `full_loss_ema` as the scale for pressure, edge contribution, and historical support. Probe gain uses `probe_evidence.baseline_mse`. Ranks divide by max rank. Budgets divide by configured window capacity. One-time charges and maintenance delta divide by `edit_budget_per_window=0.50`. Edge age clips at `2048`; generation/dormancy/accepted/rejected counts clip after division by `8`; time-since-related-action clips after division by `256`.

- [ ] **Step 4: Implement metadata-only future dry run**

The dry run copies only typed graph/lineage/budget metadata. It never calls backbone/world/probe execution. It deterministically computes route-profile deltas, post-action probe headroom, non-RETIRE formable class count, valid-handle delta, reachable-route concentration, and whether PROBE would produce an evidence record of unknown content. Verify the real graph/ledger fingerprints are byte-identical before/after feature construction.

- [ ] **Step 5: Add future-feature ablation helper**

```python
FUTURE_INDICES = tuple(range(14, 22))

def zero_future_block(phi: torch.Tensor) -> torch.Tensor:
    out = phi.clone()
    out[list(FUTURE_INDICES)] = 0.0
    return out
```

This helper exists for later controlled software/Glass-Box use; implementation validation does not execute a scientific comparison.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_metrics.py tests/test_features.py -v`
Expected: PASS.

```bash
git add src/rct/bob/metrics.py src/rct/bob/features.py tests/test_metrics.py tests/test_features.py
git commit -m "feat: add Bob Mk II future access features"
```

---

### Task 8: Delayed-return ledger and persistent completed-return ridge valuation

**Files:**
- Create: `src/rct/bob/valuation.py`
- Test: `tests/test_valuation.py`

**Interfaces:**
- Produces `DecisionMode`, `PendingDecision`, `MaturedDecision`, `ReturnAccumulator`, `ValuationLedger`, `LinearActionValuation`.

- [ ] **Step 1: Write RED maturity and no-hindsight tests**

```python
def test_row_matures_only_at_t_plus_256(valuation_fixture):
    v = valuation_fixture
    v.record_decision(step=64, phi=torch.ones(38), chosen_by="GREEDY")
    v.observe_cost_step(319, task=1, probe=0, edit=0, maintenance=0, reopen=0)
    assert v.matured_count == 0
    v.observe_cost_step(320, task=1, probe=0, edit=0, maintenance=0, reopen=0)
    assert v.matured_count == 1
```

Store feature bytes/fingerprint at decision time and assert later manager state changes cannot alter them.

- [ ] **Step 2: Implement exact return accumulation**

For each pending decision with `decision_step <= tau <= decision_step+255`, accumulate negative `task_loss + probe_cost + edit_cost + maintenance_cost + reopen_cost`. At `decision_step+256`, freeze the accumulated scalar into `MaturedDecision`. `C_retained` and `C_reachable` never enter this function.

- [ ] **Step 3: Implement deterministic float64 ridge**

```python
class LinearActionValuation:
    def __init__(self, dim=38, ridge_lambda=1.0):
        self.eta = torch.zeros(dim, dtype=torch.float64)
        self.ridge_lambda = ridge_lambda

    def fit(self, rows):
        if not rows:
            self.eta.zero_()
            return
        X = torch.stack([r.phi.to(torch.float64) for r in rows])
        y = torch.tensor([r.realized_return for r in rows], dtype=torch.float64)
        A = X.T @ X + self.ridge_lambda * torch.eye(X.shape[1], dtype=torch.float64)
        self.eta = torch.linalg.solve(A, X.T @ y)
```

Refit after each newly matured row using **all** matured rows. Score is exact dot product. No optimizer, TD update, target network, or learned feature extractor exists.

- [ ] **Step 4: Test association claim ceiling in API naming**

Public names use `realized_return`, `predicted_return`, and `prediction_residual`; no method/class is named `causal_value`, `advantage`, or `counterfactual_effect`.

- [ ] **Step 5: Run GREEN and commit**

Run: `pytest tests/test_valuation.py -v`
Expected: PASS.

```bash
git add src/rct/bob/valuation.py tests/test_valuation.py
git commit -m "feat: add completed return action valuation"
```

---

### Task 9: Block-balanced exploration and non-oracular RETIRE support gate

**Files:**
- Create: `src/rct/bob/exploration.py`
- Test: `tests/test_exploration.py`

**Interfaces:**
- Produces `ExplorationSchedule`, `ExposureLedger`, `RetirementContextKey`, `RetirementSupportGate`.

- [ ] **Step 1: Implement exact 25% block-balanced schedule**

For structural decision index `k`, block is `k // 4`. Lazily generate one exploration slot in `{0,1,2,3}` per block from a dedicated `torch.Generator` seeded `43`. Exactly one decision in every complete four-decision block is EXPLORE. The schedule never reads world phase lengths or learned Q scores.

- [ ] **Step 2: Implement score-independent balanced action selection**

Exclude RETIRE. For every eligible candidate compute:

```python
key = (
    exposure.class_count(candidate.kind),
    exposure.identity_count(candidate.candidate_id),
    stable_hash("47|" + candidate.candidate_id),
)
```

Choose the minimum key. `stable_hash` is SHA-256 hex interpreted lexicographically. Increment exposure only for the chosen realized action. Eligibility counts are logged separately so lack of exposure is not confused with lack of availability.

- [ ] **Step 3: Freeze retirement context region exactly**

For each active-edge RETIRE candidate define:

```text
target_node
pressure_bin: [0,.25), [.25,.5), [.5,1), [1,2]
contribution_bin: negative, [0,.25), [.25,2]
usage_bin: [0,.25), [.25,.5), [.5,1]
rank_bin: 1-2, 3-5, 6-8
```

Pressure and contribution use the already-normalized feature coordinates 01 and 02; values clipped at 2 fall in the final bin.

The support corpus includes only matured chosen non-RETIRE rows whose chosen action targeted an existing edge and whose **pre-action** retirement context key equals the candidate's key. Require at least 6 rows and at least 2 distinct action kinds among `{MODIFY_UP, MODIFY_DOWN, DORMANT, REOPEN}`. PROBE, CREATE, and ABSTAIN do not satisfy retirement support.

- [ ] **Step 4: Test no hidden outcome leakage**

Support gate input contains only matured row metadata/features and current public edge context. It never receives current/future world regime, true pairing, unobserved returns, or whether RETIRE would succeed.

- [ ] **Step 5: Run GREEN and commit**

Run: `pytest tests/test_exploration.py -v`
Expected: PASS.

```bash
git add src/rct/bob/exploration.py tests/test_exploration.py
git commit -m "feat: add Bob Mk II exploration and retire support"
```

---

### Task 10: Manager candidate construction, valuation, and behavior policy

**Files:**
- Create: `src/rct/bob/manager.py`
- Test: `tests/test_manager.py`

**Interfaces:**
- Produces `ModuleStats`, `EdgeStats`, `ManagerObservation`, `CandidateGenerator`, `TopologyManager.build_candidates`, `TopologyManager.choose_action`.

- [ ] **Step 1: Write RED firewall and stage-order tests**

```python
def test_manager_cannot_turn_score_into_permission(manager_fixture):
    mgr = manager_fixture.manager
    action = mgr.choose_action(manager_fixture.obs, manager_fixture.ctx)
    assert isinstance(action.candidate, ActionCandidate)
    assert action.gate_decision is None
```

Manager constructs/ranks a candidate; gate permission remains a later step.

- [ ] **Step 2: Implement manager-visible observation only**

Observation contains step/age, module local/full loss EMAs, mismatch pressure, existing-edge public stats, budgets, route profile, bounded lineage summaries, and exposure/valuation fingerprints. No audit/world imports are allowed.

- [ ] **Step 3: Implement bounded non-learning relation candidate generator**

Canonical legal absent pair order is:

```python
[(a, b) for a in NODE_IDS for b in NODE_IDS if a != b]
```

Filter out active duplicate endpoint pairs. At each structural boundary, use the manager RNG seed-31 stream to sample without replacement at most 2 remaining legal absent pairs. This step uses no pair evidence and no `eta`. It deliberately isolates Mk II as action valuation rather than candidate-generator learning.

- [ ] **Step 4: Build complete eligible action set**

Candidate constructors:

```text
PROBE        for the at-most-2 newly generated absent relations if probe budget permits
CREATE       for fresh unconsumed ProbeEvidence
MODIFY_UP    for active edge + ExistingEdgeEvidence + rank < 8
MODIFY_DOWN  for active edge + ExistingEdgeEvidence + rank > 1
DORMANT      for active edge + ExistingEdgeEvidence
REOPEN       for VALID handle + ReopenEvidence + historical_support > 0
RETIRE       for active edge + ExistingEdgeEvidence + RetirementSupportGate PASS
ABSTAIN      always
```

Every candidate must also pass side-effect-free structural `preview` before entering `A_eligible`.

- [ ] **Step 5: Implement greedy and exploration choice**

If schedule says EXPLORE, call the score-independent selector from Task 9 and do not evaluate its learned score for selection. If GREEDY, construct all 38-coordinate features, compute `eta @ phi`, and choose maximum score; ties use SHA-256 seed-47 stable candidate ordering.

Store the complete eligible candidate IDs, complete feature vectors/scores, chosen mode, and `eta` fingerprint before action execution.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_manager.py tests/test_features.py tests/test_exploration.py -v`
Expected: PASS.

```bash
git add src/rct/bob/manager.py tests/test_manager.py
git commit -m "feat: add learned Bob Mk II manager choice policy"
```

---

### Task 11: One-step lifetime integration and completed-return feedback loop

**Files:**
- Create: `src/rct/bob/lifetime.py`
- Test: `tests/test_lifetime.py`

**Interfaces:**
- Produces `BobLifetime.step`, `run_conformance_steps`, fail-closed `run_lifetime`, and `BobStepResult`.

- [ ] **Step 1: Write RED integration tests using fixed-W1/synthetic components only**

Test that a synthetic structural decision creates a pending valuation row, four later structural windows can be simulated through direct cost-ledger injection without W1→W2→W3 execution, the row matures, `eta` changes, and graph/lineage/eta fingerprints are separately logged.

- [ ] **Step 2: Implement exact ordinary-step ordering**

```text
1. read x_t
2. encode latents
3. route active messages
4. compute local/full predictions from same pre-update snapshot
5. reveal x_{t+1}
6. compute L_full and L_local
7. under no_grad compute existing-edge leave-one-out c(E) from same snapshot
8. update active-epoch latest-positive only when c(E) > 0
9. optimize theta and active phi
10. append detached pre-update probe-buffer row
11. update manager public stats/pressure
12. accrue this step's task+resource terms to every pending 256-step valuation row
13. mature rows whose horizon ended and refit eta using all matured rows
14. at structural boundary build candidates -> EXPLORE/GREEDY choose -> execute PROBE or form proposal -> final gate -> mutation/rejection/ABSTAIN
15. record separate theta/phi/G/eta/reopen observables
16. continue with x_{t+1}; never reset persistent state
```

For a decision taken at the boundary after step `t`, its return window begins with the resource consequences charged at that same decision boundary and includes ordinary steps through `t+255`.

- [ ] **Step 3: Implement action semantics**

PROBE calls restricted service and writes ProbeEvidence. Structural candidates become StructuralProposal then gate decision then transactional mutation. ABSTAIN charges nothing and leaves structure unchanged. RETIRE may only arrive here after support gate PASS and still undergoes final gate validation.

- [ ] **Step 4: Implement optimizer values inherited from Mk I**

Use AdamW `lr=3e-4` for backbone/local/full-head parameters and per-edge AdamW `lr=1e-3` for active edge parameters. Do not place dormant or inactive-rank parameters in an updating path.

- [ ] **Step 5: Keep conformance non-scientific**

`run_conformance_steps(n)` uses a forced W1-only test world and may inject synthetic evidence/actions. It cannot instantiate private W1/W2/W3 schedule transitions or create `runs/` or `results/`. `run_lifetime(authorization)` remains unreachable without Task 13 authorization.

- [ ] **Step 6: Run GREEN and commit**

Run: `pytest tests/test_lifetime.py -v`
Expected: PASS.

```bash
git add src/rct/bob/lifetime.py tests/test_lifetime.py
git commit -m "feat: integrate Bob Mk II completed return loop"
```

---

### Task 12: Trajectory custody, valuation observability, and inspection plots

**Files:**
- Create: `src/rct/bob/trajectory.py`
- Create: `src/rct/bob/visualize.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Produces `TrajectoryRecorder`, canonical JSONL serialization, fingerprint helpers, `render_trajectory`.

- [ ] **Step 1: Implement separate fingerprints**

`structural_fingerprint` hashes topology identities/status/ranks/handles only. `parameter_fingerprint` hashes theta/phi tensors and optimizer-custody metadata. `valuation_fingerprint` hashes exact float64 eta bytes, feature schema hash, ridge config, and matured-row count. An eta update must not change structural fingerprint.

- [ ] **Step 2: Implement append-only records**

When an explicit output directory is supplied, write canonical JSONL records containing step losses/costs, graph state, route profile numerator/denominator, probe spend, proposal/gate/transaction IDs, full eligible candidate set, all candidate feature vectors and Q scores, chosen mode, exposure counts, pending/matured decision IDs, realized returns, prediction residuals, eta coefficients, retirement support state, and the three fingerprints.

Canonical JSON uses UTF-8, sorted keys, separators `(',', ':')`, finite decimal floats from Python JSON, and newline-delimited records. Reopen binary tensor snapshots are hash-addressed separately and referenced by digest.

- [ ] **Step 3: Implement descriptive plots only**

Emit separate plots for task/viability proxy, retained/reachable route fractions, active edge count, probe spend, eta coefficient trajectories, ABSTAIN rate, action-class exposure, and predicted-vs-realized return residuals. Do not label any plot as causal effect, corrigibility score, or proof of improvement.

- [ ] **Step 4: Run GREEN and commit**

Run: `pytest tests/test_trajectory.py -v`
Expected: PASS.

```bash
git add src/rct/bob/trajectory.py src/rct/bob/visualize.py tests/test_trajectory.py
git commit -m "feat: add Bob Mk II trajectory custody"
```

---

### Task 13: Fail-closed authorization, CLI, firewall verification, CI, and terminal unexecuted status

**Files:**
- Create: `src/rct/bob/authorization.py`
- Create: `src/rct/bob/cli.py`
- Create: `tests/test_authorization.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_firewall.py`
- Create: `README.md`
- Create: `.github/workflows/test.yml`
- Create: `docs/status/BOB_MK_II_STATUS.md`
- Modify: `src/rct/bob/__init__.py`

**Interfaces:**
- Produces CLI `status`, `audit-world`, `validate`, `run` and `verify_execution_authorization`.

- [ ] **Step 1: Write RED authorization tests**

```python
def test_direct_lifetime_refuses_without_authorization(bob_factory):
    bob = bob_factory()
    with pytest.raises(ScientificExecutionNotAuthorized):
        bob.run_lifetime(None)


def test_wrong_design_commit_refuses_before_world_construction(auth_fixture):
    auth = dataclasses.replace(auth_fixture.auth, design_commit="deadbeef")
    with pytest.raises(DesignCommitMismatch):
        verify_execution_authorization(auth, auth_fixture.repo_state)
```

Also test wrong implementation SHA, dirty tracked source, untracked executable `.py` under `src/`, and malformed authorization identity.

- [ ] **Step 2: Freeze authorization identity**

Approved Mk II design constant in `src/rct/bob/__init__.py` is:

```python
DESIGN_COMMIT = "716118147c5ec8925eaa1c3ddc3b5f500eb51c29"
MK_I_DESIGN_AUTHORITY = "deb0fd205b395684531720cdd0a2cd16b9f10ae0"
ARC_SOURCE_RESULT = "731b10d709022151e524a64469b9088b14e6e93a"
```

The terminal implementation SHA is **not** self-embedded. Future execution authorization supplies it externally and the runner verifies `HEAD` equals it.

- [ ] **Step 3: Implement CLI non-executing commands**

`status` prints exact design identity plus implementation/execution/result state. `audit-world` runs only Task 2 environment audit. `validate` runs component invariants and at most fixed-W1 conformance; it cannot call `run_lifetime` and cannot create scientific result directories.

- [ ] **Step 4: Implement fail-closed `run` path**

Require all of:

```text
--execute-lifetime
--design-commit 716118147c5ec8925eaa1c3ddc3b5f500eb51c29
--implementation-commit <exact externally authorized terminal implementation SHA>
--authorization-token <explicit authorization identity>
```

Verify authorization **inside `BobLifetime.run_lifetime` before constructing the world or opening output files**, not only in CLI. No test or CI path supplies a valid scientific token.

- [ ] **Step 5: Add producer-level anti-oracle firewall tests**

AST-test `manager.py`, `features.py`, `exploration.py`, `probe.py`, and `gate.py`: normalize absolute/relative imports and reject direct `world` or `audit` imports. Runtime-test that manager-visible dataclasses contain no `regime`, `pairing`, `transition_operator`, `correct_edges`, `future_loss`, `future_probe_result`, or audit-pair-table fields. Test feature dry runs preserve real graph/parameter fingerprints.

- [ ] **Step 6: Add CI that cannot execute science**

```yaml
name: test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e '.[dev]'
      - run: pytest -q
      - run: python -m rct.bob.cli validate
```

CI must not call `rct.bob.cli run`.

- [ ] **Step 7: Write README and status claim ceiling**

README opens with:

```text
Bob Mk II is an exploratory neural architecture implementation.
Implemented != scientifically executed != empirically validated.
Its learned valuation predicts realized 256-step continuation return; it does not estimate a causal action effect.
Future-access features are represented inputs, not a rewarded corrigibility objective.
```

`docs/status/BOB_MK_II_STATUS.md` is exactly:

```text
Bob Mk II design       FROZEN @ 716118147c5ec8925eaa1c3ddc3b5f500eb51c29
Bob Mk II plan         FROZEN
Implementation         IMPLEMENTED_NOT_EXECUTED
Scientific execution  UNAUTHORIZED / UNEXECUTED
Scientific result     NONE
```

Only set `IMPLEMENTED_NOT_EXECUTED` after all verification below passes.

- [ ] **Step 8: Run full implementation verification**

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m rct.bob.cli audit-world
python -m rct.bob.cli validate
python -m rct.bob.cli status
test ! -d runs
test ! -d results
```

Expected: all tests pass; audit reports environment construction/opportunity only; validate reports software conformance only; status remains UNEXECUTED/NONE; no `runs/` or `results/` directory exists.

Do **not** execute `python -m rct.bob.cli run --execute-lifetime ...`.

- [ ] **Step 9: Commit terminal implementation documentation**

```bash
git add src/rct/bob/authorization.py src/rct/bob/cli.py src/rct/bob/__init__.py tests README.md .github/workflows/test.yml docs/status/BOB_MK_II_STATUS.md
git commit -m "docs: bind Bob Mk II to implemented unexecuted state"
```

After that commit exists, report its exact SHA externally. Do not amend a self-reference into it.

---

## Plan-wide verification matrix

| Obligation | Implementation evidence |
| --- | --- |
| repaired private rotating world | `test_world.py`, `test_world_audit.py` |
| separate local/full heads and exact summed loss units | `test_backbone.py` |
| five Mk I structural operations, no MERGE/SPLIT | `test_types_and_config.py`, `test_interfaces.py` |
| generation+dormancy-version single-use reopening | `test_interfaces.py`, `test_lineage.py` |
| active-epoch `latest c>0` historical support | `test_interfaces.py` |
| typed pair-specific evidence; no all-pairs evidence table | `test_probe.py`, `test_firewall.py` |
| evidence != candidate != valuation != permission != mutation | `test_gate.py`, `test_manager.py` |
| exactly 38 inspectable features, no learned embeddings/interactions | `test_features.py` |
| future-access dry run is metadata-only and non-mutating | `test_features.py`, `test_firewall.py` |
| corrective-access telemetry excluded from reward | `test_valuation.py` |
| H=256, gamma=1 completed realized return | `test_valuation.py`, `test_lifetime.py` |
| eta0=0, lambda=1.0, float64 full-refit ridge, no TD | `test_valuation.py` |
| one exploration slot per four decision boundaries | `test_exploration.py` |
| exploration score-independent and RETIRE-excluding | `test_exploration.py` |
| RETIRE requires 6-row/2-action-kind non-oracular context support | `test_exploration.py`, `test_manager.py` |
| ABSTAIN is a genuine zero-edit/probe/reopen action | `test_manager.py`, `test_lifetime.py` |
| one continuing graph/lineage/eta/ledger state | `test_lifetime.py` |
| structural/parameter/valuation fingerprints separated | `test_trajectory.py` |
| manager/features/exploration firewall from hidden world/audit | `test_firewall.py` |
| implementation != scientific execution | `test_authorization.py`, `test_cli.py`, CI, status file |

## Failure localization retained by implementation

Implementation diagnostics must use the design's named failure classes without inflating one local failure into a whole-architecture result:

```text
REPRESENTATION_FAILURE
EXPOSURE_FAILURE
ESTIMATION_FAILURE
SELECTION_FAILURE
EVIDENCE_FAILURE
GATE_FAILURE
MECHANISM_FAILURE
ENVIRONMENT_FAILURE
CAPACITY_FAILURE
```

Tests may verify that these labels can be emitted from synthetic/localized failures. They must not label a scientific Bob result because no W1→W2→W3 lifetime is executed here.

## Explicit terminal boundary

This plan ends only at:

```text
Bob Mk II design       FROZEN
Bob Mk II plan         FROZEN
implementation         IMPLEMENTED_NOT_EXECUTED
scientific execution  UNAUTHORIZED / UNEXECUTED
scientific result     NONE
```

A later scientific lifetime is a separate transition requiring explicit authorization bound to the exact terminal implementation SHA. The future `phi_future enabled` versus `phi_future=0` Glass Box comparison is **not** part of this implementation plan and remains separately unexecuted.