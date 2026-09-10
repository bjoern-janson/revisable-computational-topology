# Bob Mk II — Learned Inquiry and Structural Action Valuation

**Repository:** `bjoern-janson/revisable-computational-topology`

**Branch:** `design/bob-mk-ii-inquiry-valuation`

**Exact parent:** `37f8b3df554ddffcfdf4b17a51b3fd703294ae78`

**Inherited Bob Mk I design authority:** `deb0fd205b395684531720cdd0a2cd16b9f10ae0`

**Arc Reactor Mk II source-result record:** `bjoern-janson/arc-reactor-mk-ii@731b10d709022151e524a64469b9088b14e6e93a`

**Standing:** `DESIGN_WRITTEN_FOR_REVIEW / IMPLEMENTATION_NOT_AUTHORIZED / SCIENTIFIC_EXECUTION_UNAUTHORIZED`

This document defines Bob Mk II as a new architectural generation. It does not retroactively modify Bob Mk I, does not import Arc Reactor evidence into Bob, and does not authorize implementation or a W1→W2→W3 scientific lifetime.

---

## 0. Program boundary

Bob Mk II inherits the repaired Bob Mk I organism, world, structural vocabulary, typed evidence boundary, authoritative gate, generation-safe reopening lifecycle, optimizer semantics, route-level corrective-access telemetry, and Bob→Glass Box evidential separation.

Mk II adds one new object:

```math
\boxed{\text{a persistent learned valuation over currently available inquiry and structural actions}.}
```

The motivating source result is narrow. Arc Reactor Mk II established, in its own prospectively frozen finite apparatus, that terminal feedback could train a compact persistent selector to choose present inquiries partly according to downstream diagnostic structure. Bob Mk II imports only that **candidate architectural motif**. Arc does not establish that the motif will work in Bob, that its learned coefficients transfer, or that future-access-sensitive valuation improves neural topology revision.

Freeze:

```math
\boxed{\text{shared architectural motif}\neq\text{shared evidence}.}
```

Mk II's north-star question is:

```math
\boxed{\textbf{Can a continuing neural organism learn which inquiry or structural action is worth taking partly because of the future diagnostic and corrective options that action leaves available?}}
```

---

## 1. Exact inheritance from Bob Mk I

Mk II does not reopen the following Mk I decisions:

- fixed endpoint vocabulary `A,B,C,D`;
- structural vocabulary exactly `CREATE`, `MODIFY`, `DORMANT`, `RETIRE`, `REOPEN`;
- `MERGE` and `SPLIT` absent;
- one continuing lifetime with no reset of graph, lineage, manager state, or learned valuation state across W1/W2/W3;
- fixed-topology neural backbone plus mutable directed low-rank adapter graph;
- separate local and full predictive heads;
- summed coordinate-mean MSE task/economic units;
- temporary read-only `RELATION_PROBE` with lifetime-accounted cost;
- typed `ProbeEvidence`, `ExistingEdgeEvidence`, and `ReopenEvidence`;
- authoritative non-oracular structural gate;
- edge identity `(source,target,generation)`;
- dormancy-versioned single-use handles with atomic parameter+optimizer snapshots;
- active-epoch `historical_support = latest c(E) > 0`, or `NONE`;
- route-level `C_retained` and `C_reachable` as descriptive telemetry only;
- scientific execution fail-closed on exact design + exact authorized implementation identity.

Mk II therefore changes **manager action valuation**, not the meaning of structural legality or evidence.

---

## 2. New organism state and adaptation seam

Mk II extends the continuing organism state with a learned valuation parameter vector and an explicit delayed-return ledger:

```math
\boxed{X_t^{II}=(\theta_t,\mathcal G_t,\Lambda_t,\sigma_t,\eta_t,\mathcal D_t).}
```

where:

- `theta_t` — ordinary backbone/head parameters;
- `G_t` — current mutable adapter topology and edge parameters;
- `Lambda_t` — immutable structural lineage and reopening substrate;
- `sigma_t` — manager history, pressure, costs, candidate/exposure state;
- `eta_t` — persistent learned action-valuation coefficients;
- `D_t` — pending and matured valuation-training records.

Freeze the attribution seam:

```math
\boxed{\theta\text{-plasticity}\neq\phi\text{-plasticity}\neq\mathcal G\text{-plasticity}\neq\eta\text{-plasticity}\neq\mathrm{REOPEN}.}
```

A later Bob observation may show these channels co-occurring. It does not identify which one caused a performance change.

---

## 3. Candidate generation, evidence, valuation, permission, mutation

Mk II preserves the ordering:

```text
current consequences
-> candidate generation
-> typed evidence / non-mutating eligibility construction
-> learned valuation
-> proposal
-> authoritative gate
-> mutation or rejection
```

and the constitutive distinction:

```math
\boxed{\text{evidence}\neq\text{candidate}\neq\text{valuation}\neq\text{proposal}\neq\text{permission}\neq\text{mutation}.}
```

The learned valuation may rank only actions that the current public manager state can legitimately instantiate as candidates. It may never manufacture missing evidence, revive a consumed handle, bypass budget/cooldown/rank/generation constraints, or override the gate.

A non-mutating eligibility preview may derive from the same authoritative state used by the gate. Preview means only `currently formable if selected`; it is not permission and may become stale. The chosen structural action must be revalidated by the authoritative gate immediately before mutation.

### 3.1 Action-choice universe

At a structural decision boundary the candidate action set may contain:

```math
\mathcal A_t^{eligible}\subseteq\{
\mathrm{PROBE}(i\to j),
\mathrm{CREATE}(i\to j),
\mathrm{MODIFY}(E,\pm1),
\mathrm{DORMANT}(E),
\mathrm{RETIRE}(E),
\mathrm{REOPEN}(H),
\mathrm{ABSTAIN}
\}.
```

`ABSTAIN` is always available, has zero probe/edit charge, and leaves persistent structure unchanged for that decision boundary.

Prerequisites remain typed:

```text
PROBE          -> system-generated endpoint relation + probe budget
CREATE         -> valid ProbeEvidence for exact endpoints
MODIFY         -> valid ExistingEdgeEvidence for exact edge generation
DORMANT        -> valid ExistingEdgeEvidence for exact edge generation
RETIRE         -> valid ExistingEdgeEvidence + separate retirement-support gate
REOPEN         -> valid ReopenEvidence for exact VALID handle/version
ABSTAIN        -> no evidence prerequisite
```

The manager may still make bad choices. Legality is not scientific correctness.

---

## 4. Shared linear valuation model

Mk II uses one shared inspectable linear action-value model:

```math
\boxed{Q_\eta(s_t,a)=\eta_t^\top\phi(s_t,a).}
```

There is no per-operation neural value head and no hidden planner. The same coefficient vector scores all action classes in a common feature space.

The feature schema must explicitly separate:

```math
\boxed{\phi=[\phi_{now},\phi_{cost},\phi_{future},\phi_{lineage},\phi_{action}].}
```

### 4.1 `phi_now`

May contain only present manager-observable quantities such as:

- current target/edge mismatch pressure;
- current edge contribution and usage for existing-edge actions;
- current probe evidence magnitude/age where such evidence already exists;
- remaining structural/probe budget;
- current rank and legal rank direction;
- current uncertainty/dispersion summaries already exposed by the Mk I manager interface.

It may not contain hidden regime identity, correct-edge labels, audit-private pair scores, or future task outcomes.

### 4.2 `phi_cost`

Contains authoritative current charges or liabilities visible to the manager, including as applicable:

- probe cost;
- edit cost;
- maintenance delta implied by the candidate;
- reopening liability;
- remaining budget fraction.

Proposal-supplied cost guesses are not features when authoritative costs exist.

### 4.3 `phi_future`

This is the new scientific seam. It may describe **future structural/inquiry access implied by current public state and the deterministic bookkeeping consequences of the candidate**, but may not inspect future world observations.

Allowed one-step structural dry-run quantities include:

- change in number/fraction of currently retained historical correction routes;
- change in number/fraction of currently reachable historical correction routes under present budget/lifecycle rules;
- number of remaining unspent relation-probe opportunities after the action;
- number of currently formable non-RETIRE candidate classes after deterministic application of the action's declared bookkeeping consequences;
- whether an action creates, consumes, or revokes a valid reopening handle;
- whether a candidate concentrates future access into a single remaining route versus leaving multiple presently represented routes.

The dry run is metadata-only. It may not execute a neural/world rollout, query the hidden transition operator, observe a future probe result, use the true post-action task loss, or invoke an exact planner over future world trajectories.

Freeze:

```math
\boxed{\text{represented future-access features}\neq\text{oracle future value}.}
```

`C_retained` and `C_reachable` may therefore appear as **inputs or deltas** in `phi_future`; they do not appear as direct reward terms. This allows Bob to learn whether preserving those represented options predicts realized later value rather than being paid merely to maximize the telemetry.

### 4.4 `phi_lineage`

May contain public lineage facts such as:

- edge age;
- current generation and dormancy version;
- handle state;
- active-epoch historical support;
- time since last related structural action;
- bounded counts of prior accepted/rejected actions of the same typed relation;
- whether a route has previously been exercised in a later correction episode.

Historical success remains evidence about history, not current truth.

### 4.5 `phi_action`

Operation identity is explicit. The implementation plan must freeze an inspectable encoding for `PROBE`, `CREATE`, `MODIFY_UP`, `MODIFY_DOWN`, `DORMANT`, `RETIRE`, `REOPEN`, and `ABSTAIN` plus any required interaction terms. It may not introduce a hidden learned embedding or extra neural network without reopening the design.

---

## 5. Delayed realized return

The manager is not trained on immediate reward alone and is not directly rewarded for corrective-access telemetry.

For a structural decision taken at ordinary step `t`, freeze the training horizon:

```math
\boxed{H=256\text{ ordinary steps}=4\text{ structural windows}.}
```

Use undiscounted finite-horizon realized return (`gamma = 1`):

```math
\boxed{
Y_t=-\sum_{\tau=t}^{t+255}
\left[
L_{task}(\tau)
+C_{probe}(\tau)
+C_{edit}(\tau)
+C_{maintenance}(\tau)
+C_{reopen}(\tau)
\right].
}
```

The exact Mk I task/economic units carry through unchanged.

`C_retained` and `C_reachable` are excluded from `Y_t`.

Thus a `DORMANT` action is valuable only if its actual later consequences justify its costs relative to alternatives under Bob's continuing policy. A `RETIRE` action may receive an immediate maintenance advantage but can suffer later if revoking reopening routes makes subsequent recovery expensive. Nothing in the reward states that preserved corrigibility is intrinsically good.

### 5.1 Claim ceiling of the target

`Y_t` is an observed finite-horizon consequence under Bob's actual continuation process. It is **not** a matched counterfactual effect of action `a_t`, because later actions, learning, and world events also contribute to the realized return.

Freeze:

```math
\boxed{\text{realized delayed association used for learning}\neq\text{causal action effect}.}
```

A causal mechanism claim requires a later matched Glass Box assay.

---

## 6. Delayed-return ledger and learning rule

At each structural decision, Bob records an immutable pending valuation row containing at least:

```text
decision_id
decision_step
manager-state fingerprint
complete eligible-action identity set
chosen action identity
chosen_by = EXPLORE | GREEDY
action feature vector at decision time
eta identity at decision time
relevant evidence/handle/generation references
maturity_step = decision_step + 256
```

The feature vector is frozen at decision time. It may not be recomputed later using hindsight.

When `maturity_step` is reached, the row receives its realized `Y_t` and becomes matured. Unmatured rows never train `eta`.

Mk II uses completed-return regularized least squares, not temporal-difference bootstrapping:

```math
\boxed{
\eta_k=\arg\min_\eta
\sum_{i\in\mathcal D_k^{mature}}
(Y_i-\eta^\top\phi_i)^2
+\lambda_\eta\|\eta-\eta_0\|_2^2.
}
```

Freeze:

```math
\eta_0=0.
```

There is no learned bootstrap target, target network, Bellman backup, or value estimate inside another value target.

The exact positive numerical `lambda_eta`, feature normalization constants, and numerical solver are implementation-plan constants. They must be frozen before implementation/scientific execution and may not be selected using W1→W2→W3 scientific outcomes.

Exploratory and greedy rows may both train the shared regressor because both are real interventions in the continuing lifetime. Their provenance remains separately reportable.

Overlapping 256-step returns are allowed as training records but may not be interpreted as statistically independent scientific replications.

---

## 7. Exploration and exposure provenance

Mk II must not train only from actions its current greedy valuation already prefers.

A prospectively fixed fraction of structural decision boundaries is assigned to score-independent exploration using an RNG stream independent of the world schedule. Conditional on the currently eligible non-RETIRE set, exploratory choice is independent of current `Q_eta` scores.

The implementation plan must freeze before any scientific execution:

- exploration fraction/schedule;
- independent RNG seed/stream;
- deterministic balancing rule across currently eligible action classes and identities;
- tie-breaking;
- behavior when only `ABSTAIN` is eligible.

The balancing rule should minimize exposure-count imbalance among actions/classes that were actually eligible; it must not claim equal exposure to actions that the evolving state never made eligible.

Every exploratory action pays ordinary real costs and has ordinary real consequences. There is no free training sandbox inside the lifetime.

### 7.1 RETIRE is never forced

`RETIRE` is excluded from score-independent forced exploration.

A retirement candidate may enter greedy competition only after a prospectively frozen **retirement support gate** judges the candidate's non-oracular feature region sufficiently supported by matured non-RETIRE experience. The support gate exists to prevent the first irreversible action from being selected solely from unconstrained extrapolation.

The implementation plan must freeze the support representation, minimum support count, neighborhood/coverage rule, and failure behavior before implementation. The support test may use only matured valuation rows and current public features; it may not use hidden-world correctness or future outcomes not yet observed.

Failure of the support gate means `RETIRE` is absent from `A_t^eligible` for that decision. This is a safety/identifiability design constraint, not evidence that retirement would have been bad.

---

## 8. Behavior policy

At a normal greedy structural decision:

```math
\boxed{a_t=\arg\max_{a\in\mathcal A_t^{eligible}}Q_{\eta_t}(s_t,a)}
```

with prospectively frozen deterministic tie-breaking.

At an exploration-assigned decision, the frozen balancing policy selects among currently eligible non-RETIRE actions without consulting their learned scores.

`ABSTAIN` competes in both modes where the balancing rule permits it. It has a learned predicted return like any other action. Mk II can therefore learn that spending or editing is not worthwhile in a particular observed state.

The behavior policy cannot enlarge the candidate universe or bypass typed evidence.

---

## 9. Special semantics by action class

### 9.1 PROBE

A probe action buys one Mk I `RELATION_PROBE` for a relation first generated by Bob. Its immediate persistent topology effect is none; its deterministic future-access feature effects may include reduced probe budget and the possibility that a typed evidence record will exist after execution. The feature constructor may not know the probe answer before selection.

### 9.2 CREATE

`CREATE` may be valued only with valid exact-endpoint `ProbeEvidence`. Future-access features may inspect deterministic bookkeeping/topology consequences of adding the proposed edge, but not how well its newly initialized neural parameters will eventually perform.

### 9.3 MODIFY

Rank-up/rank-down candidates require valid existing-edge evidence and exact generation/rank legality. Ordinary phi gradient learning is never represented as a structural MODIFY action.

### 9.4 DORMANT

The feature constructor may represent the deterministic creation of a new dormancy-version handle, maintenance reduction, and currently represented retained/reachable-route consequences. It may not assume that the handle will later prove useful.

### 9.5 REOPEN

Only an exact VALID handle with typed ReopenEvidence may be scored. Its feature vector may contain historical support and liability, but historical support remains merely `useful then`. Successful REOPEN consumes the handle exactly as in Mk I.

### 9.6 RETIRE

RETIRE remains the only structural action explicitly excluded from forced exploration because it revokes cheap-revival handles for the generation. It still requires normal Mk I legality, ExistingEdgeEvidence, and the new retirement support gate. A high learned score never overrides these prerequisites.

### 9.7 ABSTAIN

ABSTAIN makes no structural edit, buys no probe, and pays no edit/probe/reopen charge at that decision. Ordinary neural learning and ordinary maintenance of already-active structure continue during the subsequent horizon.

---

## 10. Observability and custody

Mk II adds at least these descriptive records to the Mk I trajectory:

```text
eta trajectory / coefficient snapshots
candidate-set and eligibility trajectory
per-candidate feature vectors and Q scores
EXPLORE vs GREEDY provenance
pending/matured delayed-return rows
return-prediction residuals
per-action-class exposure counts
ABSTAIN frequency
RETIRE support-gate states
future-access feature values/deltas
```

The scientific record must preserve the exact feature schema, normalization, regression configuration, exploration configuration, and action tie-breaking used by the implementation.

Separate structural fingerprints, parameter fingerprints, and valuation-state fingerprints. A change in `eta` is not a topology mutation.

---

## 11. Primary exploratory comparisons

Bob Mk II remains an exploratory organism rather than a predeclared single-score benchmark. Nevertheless the architecture is deliberately constructed to support later causal assays.

The especially important later Glass Box contrast is:

```math
\boxed{\phi_{future}\text{ enabled}\quad\text{vs}\quad\phi_{future}=0}
```

with candidate generation, legal actions, evidence, resource budgets, training target, and other features held matched.

That later assay could ask whether future-access-sensitive valuation causally changes:

- action selection;
- realized task/resource return;
- post-shift recovery;
- DORMANT versus RETIRE choices;
- later reopening use;
- retained/reachable route trajectories.

This design does not itself authorize or preregister that Glass Box execution.

---

## 12. Failure localization

A Mk II null must be localized rather than inflated into a whole-program failure. Distinguish at least:

```text
REPRESENTATION_FAILURE
  phi lacks information needed to rank useful actions

EXPOSURE_FAILURE
  relevant action/state regions never receive informative mature outcomes

ESTIMATION_FAILURE
  linear ridge valuation misestimates returns despite adequate representation/exposure

SELECTION_FAILURE
  estimated ranking is adequate but behavior/exploration policy does not use it effectively

EVIDENCE_FAILURE
  useful structural candidates cannot become typed/eligible

GATE_FAILURE
  legal candidate cannot pass the intended non-oracular contract

MECHANISM_FAILURE
  selected action executes but does not produce the expected structural consequence

ENVIRONMENT_FAILURE
  world does not supply the intended learnable/revisable relational opportunity

CAPACITY_FAILURE
  downstream theta/phi adaptation cannot exploit an otherwise useful structural action
```

A null at one layer does not establish failure of every other layer.

---

## 13. Explicit non-claims

Bob Mk II does not claim in advance that:

- learned action valuation improves over Bob Mk I;
- future-access features are useful;
- the linear model is sufficient;
- delayed realized return identifies causal action value;
- the manager discovers a universal corrigibility objective;
- `C_retained` or `C_reachable` is a sufficient measure of correction;
- RETIRE support makes irreversible action safe;
- Bob invents new query semantics, endpoint ontologies, modules, or interface programs;
- Arc Reactor's learned coefficients transfer to Bob;
- Bob performs general active learning, neural architecture search, recursive self-improvement, or alignment;
- whole-system recovery identifies the mechanism that caused recovery.

Even a successful Mk II lifetime remains a bounded observation in this fixed four-domain architecture and world family until separately tested.

---

## 14. Implementation and scientific-execution boundary

This design authorizes neither implementation nor scientific execution.

After written-spec review, a separate implementation plan must freeze all remaining numerical/encoding choices explicitly delegated above, including `lambda_eta`, feature normalization, exact `phi_action` encoding/interactions, exploration assignment/balancing, retirement-support metric/thresholds, tie-breaking, ledger serialization, and source/config custody.

Implementation may later terminate at:

```text
Bob Mk II design       FROZEN
Bob Mk II plan         FROZEN
implementation         IMPLEMENTED_NOT_EXECUTED
scientific execution  UNAUTHORIZED / UNEXECUTED
scientific result     NONE
```

Any eventual Mk II lifetime runner must be separately authorized against the exact frozen Mk II design and exact terminal implementation commit, while retaining the inherited Mk I fail-closed execution boundary.

---

## 15. Design summary

Bob Mk I asks whether a continuing neural organism can construct, suppress, retire, and reopen persistent computational relations under changing reality.

Bob Mk II adds:

```text
system-generated candidate
-> typed evidence / legal eligibility
-> shared learned valuation
   = current consequence
   + actual costs
   + represented future-access structure
   + lineage
-> PROBE / CREATE / MODIFY / DORMANT / RETIRE / REOPEN / ABSTAIN
-> authoritative gate
-> real consequence
-> 256-step realized task-and-resource return
-> matured regression row
-> updated persistent eta
-> future structural decision
```

The core hypothesis remains prospective:

```math
\boxed{\textbf{feedback may teach Bob not only which structure works now, but which inquiry or structural commitment leaves it better positioned to cope with what it has not yet learned.}}
```

That hypothesis is not established by this design document.