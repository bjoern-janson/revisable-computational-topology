# Bob Mk I — bounded contract-repair freeze

**Repository:** `bjoern-janson/revisable-computational-topology`

**Branch:** `freeze/bob-mk-i-repaired`

**Base:** `7996daff3269da8e6e383cccd305eee990fead1a`

**Standing:** `DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED / SCIENTIFIC_EXECUTION_UNAUTHORIZED`

This document is the normative repair overlay for Bob Mk I / Bob V0. It preserves the original design and implementation-plan artifacts at the base commit as historical records. Where this document conflicts with `docs/superpowers/specs/2026-09-07-bob-v0-design.md`, this document controls. Nothing here authorizes implementation or the W1→W2→W3 scientific lifetime.

The repair is bounded. It does not change the central RCT hypothesis, the fixed four-node endpoint vocabulary, the one-continuing-lifetime object, or the Bob→Glass Box evidence boundary.

## 1. Frozen structural vocabulary

Bob Mk I declares exactly:

```math
\boxed{\{\mathrm{CREATE},\mathrm{MODIFY},\mathrm{DORMANT},\mathrm{RETIRE},\mathrm{REOPEN}\}.}
```

`MERGE` is removed from Mk I because the one-hop forward path does not execute a serial path whose matrices could legitimately be composed. `SPLIT` remains absent. `MODIFY` is a transactional rank change within the fixed low-rank interface family; ordinary `phi` gradient updates are not `MODIFY`.

The construction ceiling remains:

> Bob Mk I may instantiate a previously absent trainable directed interface between fixed endpoints. It does not invent arbitrary module semantics, arbitrary neural programs, or a new relation ontology.

## 2. Repaired rotating-dependency world

The world uses four persistent domains `A,B,C,D` and one stationary isotropic latent process. For each active pair `(i,j)`, the regime operator uses the orthogonal self+partner block

```math
R_{pair}=\begin{bmatrix}\alpha Q & \beta Q\\-\beta Q & \alpha Q\end{bmatrix},
\qquad \alpha=0.6,\quad \beta=0.8,
\qquad \alpha^2+\beta^2=1.
```

The latent transition is

```math
z_{t+1}=\rho R_r z_t+\sqrt{1-\rho^2}\,\epsilon_t,
```

with isotropic stationary initialization and innovations. W1/W2 pair `A↔B` and `C↔D`; W3 pairs `A↔C` and `B↔D`. This preserves one-time latent marginals while keeping a stable local self-predictive term and rotating the useful partner relation.

The intended invariant comparison is W1 versus W3:

```math
\boxed{p(x_i\mid W_1)=p(x_i\mid W_3)}
```

up to the declared implementation audit tolerance, while relevant cross-time partner conditionals differ.

The manager never receives the regime identity, hidden pairing, hidden transition operator, correct edges, future loss, or audit-private pair table. W1/W2/W3 durations are privately sampled by the world; Bob may know its age but not a deterministic public switch time.

## 3. Separate local and full prediction heads

For each target domain `j`, Bob has a trained local head and a trained task-facing full head:

```math
\widehat x_j^{local}(t+1)=D_j^{local}(h_j(t)),
```

```math
\widehat x_j^{full}(t+1)=D_j^{full}(h_j(t),m_{\rightarrow j}(t)).
```

The local head receives no cross-module message. It is trained explicitly; it is not a zero-message ablation of the full decoder. This supplies a real local predictive objective and keeps the central intended signature meaningful:

```math
L_{local}\approx\text{stable},
\qquad L_{relational,old}\uparrow.
```

## 4. Frozen loss and economic unit

For observation dimension `d_x`, each target uses coordinate-mean MSE:

```math
\ell_j^{full}(t)=\frac{1}{d_x}\|\widehat x_j^{full}(t+1)-x_j(t+1)\|_2^2,
```

```math
\ell_j^{local}(t)=\frac{1}{d_x}\|\widehat x_j^{local}(t+1)-x_j(t+1)\|_2^2.
```

The four domain losses are **summed, not averaged**:

```math
L_{full}(t)=\sum_{j\in\{A,B,C,D\}}\ell_j^{full}(t),
\qquad
L_{local}(t)=\sum_{j\in\{A,B,C,D\}}\ell_j^{local}(t).
```

Training uses

```math
L_{train}=L_{full}+\lambda_{local}L_{local}.
```

A target-specific probe gain is `MSE_baseline(target) - MSE_probed(target)`. Because the aggregate task loss is a sum and only one target term changes, that numerical reduction is already in `L_full` units: there is no hidden division by four.

Existing-edge contribution uses the same unit and sign convention:

```math
c(E)=L_{full}(E\text{ masked})-L_{full}(E\text{ active}).
```

Thus `c>0` means the edge helped that measured prediction snapshot; `c<0` means it hurt.

## 5. Environment opportunity audit

Before any Bob lifetime, an environment-only audit must establish both the analytic stationary construction and an observable partner opportunity at the declared scale. The finite task-opportunity witness is a frozen ridge predictor family with

```text
ridge_lambda = 1e-3
```

and a fixed train/held-out split. One witness receives only the target's current observation; the other additionally receives the true partner's current observation. The held-out target-domain coordinate-mean MSE reduction is computed for each intended directed pair; only the declared summary/minimum enters the audit record presented outside the hidden world. The pair table remains unavailable to Bob.

This is a **finite ridge witness**, not Bayes-optimal risk and not evidence that Bob will learn the opportunity. Environment opportunity and Bob learnability remain separate.

Empirical marginal/cross-lag summaries must preserve the fact that observations are serially dependent; sequential samples may not silently be treated as iid.

## 6. Probe claim ceiling

`RELATION_PROBE(i→j)` remains temporary, read-only with respect to persistent topology/backbone, costly, and destroyed after emitting bounded typed evidence. Probe parameters may not persist as a shadow edge bank.

The rate limit is:

```math
N_{probe}(w)\le2
```

per 64-step structural window, with lifetime probe spend monotonically accumulated.

This does **not** establish that Bob cannot eventually examine all 12 directed endpoint pairs. Mk I claims only sequential paid evidence rather than a supplied all-pairs absent-edge usefulness table.

A probe is a relation-evidence heuristic. Its estimated gain is not a calibrated estimate of the realized value of a newly initialized interface after learning delay. Realized post-edit incremental value must be recorded separately.

## 7. Authoritative gate

The structural gate receives read-only authoritative views sufficient to decide legality: exact edge generation/status/rank, handle state/version/liability, typed evidence state, proposal freshness, budget/cooldown state, and already-applied proposal identity.

The gate derives charges from frozen configuration and current authoritative state. A proposal's estimated cost is never authoritative.

Freeze:

```math
\boxed{\text{evidence}\neq\text{proposal}\neq\text{permission}\neq\text{mutation}.}
```

The gate checks legality only. It does not know which hidden-world edit is scientifically correct.

## 8. Generation- and dormancy-version-safe reopening

An edge identity is

```text
(source, target, generation)
```

with generation monotone for an ordered endpoint pair.

Every `DORMANT` event within a generation receives a monotone `dormancy_version`. Its handle names the exact `(generation, dormancy_version)` and contains one atomic snapshot of the edge parameters **and the matching per-edge optimizer state**.

Handle state is exactly:

```text
VALID | CONSUMED | REVOKED
```

A successful `REOPEN` atomically restores the weight+optimizer snapshot and changes that handle from `VALID` to `CONSUMED`. Consumed handles remain historical but can never execute again. A later `DORMANT` in the same generation creates a new versioned handle. `RETIRE` revokes every still-valid handle for that generation and destroys the live optimizer state. A later `CREATE` at the same endpoints receives a new generation.

Therefore the sequence

```text
DORMANT -> H1 -> REOPEN H1 -> train -> DORMANT -> H2 -> REOPEN H1
```

must reject the final `REOPEN H1` without mutation. Only `H2` can be current and executable.

`REOPEN` remains narrower than rollback:

```math
\boxed{\mathrm{REOPEN}\neq\mathrm{RESTORE\_CHECKPOINT}.}
```

It never restores old backbone `theta`, unrelated edges, or a whole historical model state.

## 9. Active-epoch historical reopening support

Every `CREATE` or successful `REOPEN` starts a new **active epoch** for that edge generation.

During an active epoch Bob records the latest observed leave-one-out contribution satisfying exactly

```math
\boxed{c(E)>0.}
```

There is deliberately **no `0.01` cutoff** for this historical-support record. `0.01` may remain an operational consequentiality/edit threshold elsewhere; it does not define whether prior usefulness evidence exists.

When `DORMANT` occurs, the handle freezes:

```text
historical_support = latest positive contribution in this active epoch
```

or `NONE` if no positive contribution occurred in that epoch.

Historical support never carries across a successful `REOPEN` into the next active epoch. Thus an ancient positive cannot repeatedly justify reopening an edge that has remained bad through a later epoch.

The default historical `REOPEN` path requires a current `VALID` handle with `historical_support > 0`, current mismatch/endpoint pressure, sufficient budget, and ordinary gate legality. Historical support means only **this route was useful then**; it does not establish current usefulness. A handle with `historical_support = NONE` receives no default historical prior. Reopening such a route would require a separately typed fresh-evidence path in a later design.

## 10. Bob-local corrective-access telemetry

The access metric uses **typed correction routes**, not commitments, as both numerator and denominator units. A route identity contains at least:

```text
(commitment_id, route_id, operation, edge_generation, target_rank_or_handle)
```

When a structural commitment becomes consequential under the implementation's frozen rule, the then-known routes associated with that commitment enter a persistent historical route cohort `R`. Their identities remain in the denominator even if later consumed, revoked, or made unreachable.

Freeze:

```math
C_{retained}(t)=\frac{|\{r\in R:\mathrm{retained}_t(r)\}|}{|R|},
```

```math
C_{reachable}(t)=\frac{|\{r\in R:\mathrm{reachable}_t(r)\}|}{|R|}.
```

`retained` means the information/state needed for that exact route still exists and is not consumed or revoked. `reachable` means retained and presently executable under generation/lifecycle, budget, cooldown, and non-oracular gate-validity constraints. If `|R|=0`, both values are `NA`, not `1.0`.

Demonstrated correction is a **separate outcome ledger**, not a third access fraction. A correction outcome may be recorded only when a route is actually exercised after its commitment entered the cohort, during a later mismatch/revision episode, followed by the predeclared post-edit evaluation window. Initial `CREATE` does not count as demonstrated correction merely because it becomes useful.

These are descriptive Bob-local telemetry. Even `C_retained=C_reachable=1` does not establish that one bounded policy can identify and select the correct route over Bob's actual uncertainty. A jointly executable MATRIX/Glass-Box correction witness is a separate scientific object.

## 11. Optimizer lifecycle

Optimizer state is part of structural semantics:

```text
CREATE  -> fresh edge parameters + fresh per-edge optimizer state
DORMANT -> atomic frozen edge+optimizer snapshot
REOPEN  -> restore exact snapshot; consume handle; begin new active epoch
RETIRE  -> destroy live per-edge optimizer; revoke current handles
MODIFY rank down -> freeze inactive factors and matching moments
MODIFY rank up   -> resume those frozen factors and moments
```

Dormant or inactive factors must not drift because of AdamW momentum or weight decay.

## 12. Execution boundary

Implementation/conformance work may run unit tests, the environment audit, and short fixed-regime non-scientific plumbing steps. It must not execute the W1→W2→W3 Bob lifetime.

Any future lifetime run must fail closed unless it receives and verifies, **inside the runner boundary before world construction or side effects**:

```text
explicit execution authorization
exact frozen Bob Mk I design authority commit
exact authorized terminal implementation commit
clean checked-out implementation state
```

Direct Python invocation must enforce the same guard as the CLI. Authorization against only the design commit is insufficient.

## 13. Claim ceiling and lineage

Bob Mk I remains exploratory. A striking trajectory is an observation, not a mechanism result. Route availability is not jointly executable correction. Probe gain is not realized edit value. Finite endpoint-pair search is not relation-ontology invention. Whole-system recovery does not identify which adaptation channel caused recovery.

This freeze closes the bounded Mk I design-repair cycle. **No Bob Mk II change is part of this document.** Mk II must branch from the terminal Mk I documentation commit and explicitly state what it inherits and what it adds.