# Bob Mk I — repaired implementation-plan amendment

**Repository:** `bjoern-janson/revisable-computational-topology`

**Branch:** `freeze/bob-mk-i-repaired`

**Base plan:** `docs/superpowers/plans/2026-09-07-bob-v0.md` at `7996daff3269da8e6e383cccd305eee990fead1a`

**Normative repaired design authority:** `deb0fd205b395684531720cdd0a2cd16b9f10ae0`

**Standing:** `PLAN_FROZEN / IMPLEMENTATION_NOT_STARTED / SCIENTIFIC_EXECUTION_UNAUTHORIZED`

This file is the implementation-plan overlay for the exact repaired Bob Mk I design above. Where it conflicts with the base implementation plan, this amendment controls. The base plan remains a historical artifact and must not be implemented verbatim.

No implementation, unit tests, environment audit, conformance run, or W1→W2→W3 lifetime is authorized by this document.

## 1. Global implementation constraints

The implementation must preserve four separately logged adaptation channels:

```math
\boxed{\theta\text{-plasticity}\neq\phi\text{-plasticity}\neq\mathcal G\text{-plasticity}\neq\mathrm{REOPEN}.}
```

Endpoint vocabulary is exactly `A,B,C,D`. Structural vocabulary is exactly `CREATE`, `MODIFY`, `DORMANT`, `RETIRE`, `REOPEN`. `MERGE` and `SPLIT` have no Mk I constructor, gate, or application path.

Candidate generation remains bounded to the 12 directed non-self endpoint pairs. Mk I may instantiate a previously absent trainable edge; it does not invent a new endpoint or relation ontology.

## 2. Task 1 amendment — types and frozen configuration

Add or revise the typed boundary so it includes:

```text
ProbeEvidence
ExistingEdgeEvidence
ReopenEvidence
HandleState = VALID | CONSUMED | REVOKED
ExecutionAuthorization
```

`ReopenEvidence` must name exact source, target, edge generation, dormancy version, handle ID, current pressure, and `historical_support: float | None`.

Freeze these configuration semantics:

```text
loss_domain_reduction = SUM
ridge_lambda = 1e-3
probes_per_window = 2
structural_interval = 64
```

Each per-domain loss is coordinate-mean MSE. `L_full` and `L_local` are sums of the four domain losses. A target-specific probe gain or existing-edge contribution is therefore already in `L_full` units and is not divided by four.

The world samples W1, W2, and W3 durations independently from the discrete uniform distribution on each configured inclusive `[min,max]` interval using only the private world RNG. World, emission, network, manager, probe, and audit RNG streams remain separate.

Required tests include exact operation vocabulary, immutable config, exact loss reduction convention, fixed ridge lambda, and private bounded phase schedule.

## 3. Task 2 amendment — world and audit

Use the repaired self+partner orthogonal transition with `alpha=0.6`, `beta=0.8`, stationary isotropic initialization/innovation, W1/W2 pairing `AB/CD`, and W3 pairing `AC/BD`.

The environment audit must:

- analytically verify orthogonality/stationary one-time latent marginals;
- compare the intended W1 and W3 regimes, not W1 and W2, for relational change;
- keep fixed nonlinear emissions and observation noise across regimes;
- use a frozen ridge witness with `lambda=1e-3`;
- fit the witness on the first 3072 transitions and evaluate on the final 1024 of the audit sample;
- compare target-local input against target-local-plus-true-partner input;
- compute held-out target coordinate-mean MSE reduction and preserve its one-to-one conversion into summed `L_full` units;
- keep the pair table audit-private and expose only the declared summary/minimum;
- explicitly label the result as a finite predictor-family opportunity witness, not Bayes-optimal risk and not Bob learnability.

Sequential audit samples are serially dependent. Report empirical marginal/cross-lag summaries over 32 contiguous 128-transition blocks and their dispersion; do not use iid language for the 4096-point sequence.

Running this environment audit is not Bob scientific execution.

## 4. Task 3 amendment — local/full neural heads and loss equations

`decode_local(h_j)` is a separately trained local head with no incoming cross-module message. `decode_full(h_j,m_→j)` is the task-facing head.

For target `j`:

```math
\ell_j^{full}=\frac{1}{d_x}\|\widehat x_j^{full}-x_j\|_2^2,
\qquad
\ell_j^{local}=\frac{1}{d_x}\|\widehat x_j^{local}-x_j\|_2^2.
```

Then:

```math
L_{full}=\sum_j\ell_j^{full},
\qquad
L_{local}=\sum_j\ell_j^{local},
```

```math
L_{train}=L_{full}+\lambda_{local}L_{local}.
```

The full-versus-local difference is descriptive task telemetry. It is not a universal local-health or corrigibility metric.

## 5. Task 4 amendment — exact edge, handle, optimizer, and active-epoch lifecycle

Each edge is identified by:

```text
(source, target, generation)
```

with a monotone generation counter per ordered endpoint pair.

Each `DORMANT` event within a generation receives a monotone `dormancy_version`. Freeze the reopen handle structure conceptually as:

```text
handle_id
edge generation
dormancy version
atomic edge-parameter snapshot
matching per-edge optimizer-state snapshot
dormant evidence reference
historical_support: float | NONE
reopening liability
state: VALID | CONSUMED | REVOKED
```

Every `CREATE` or successful `REOPEN` starts a fresh active epoch. During that epoch, maintain only the **latest** leave-one-out contribution satisfying exactly `contribution > 0`. Zero and negative values do not qualify. There is no `0.01` cutoff for historical support.

`DORMANT` closes the active epoch and freezes its latest positive value into the handle, or `NONE` if no positive contribution occurred.

`REOPEN(H_g,n)` must atomically:

```text
verify exact generation/version and VALID state
verify endpoint/lifecycle conflict freedom
restore that handle's parameter snapshot
restore the matching optimizer snapshot
mark H_g,n CONSUMED
start a new active epoch with empty historical-support history
```

A consumed handle remains immutable lineage but cannot execute again.

The following must be an explicit RED→GREEN lifecycle test:

```text
DORMANT -> H1 -> REOPEN H1 -> train -> DORMANT -> H2 -> REOPEN H1
```

The final `REOPEN H1` must fail without graph, parameter, optimizer, budget, or ledger mutation; `REOPEN H2` may succeed if otherwise legal.

`RETIRE` destroys the live edge optimizer state, revokes every still-VALID handle for that generation, preserves historical records, and does not prevent a later same-endpoint CREATE from receiving a new generation.

`MODIFY` changes active rank by exactly one. Rank-down freezes deactivated factors and their optimizer moments; rank-up resumes those exact factors/moments. Dormant or inactive factors may not drift under AdamW momentum or weight decay.

## 6. Task 5 amendment — probe evidence and candidate boundary

`RELATION_PROBE(i→j)` remains temporary and read-only with respect to persistent topology and backbone. Temporary fitted parameters must be destroyed after emitting immutable `ProbeEvidence`; they cannot enter the persistent graph or a shadow parameter bank.

Probe allowance is at most 2 per 64-step structural window. Window allowance resets; lifetime probe expenditure never resets.

The claim is **rate-limited sequential paid evidence**, not cumulative sub-exhaustive search. Bob may eventually investigate all 12 endpoint pairs across multiple windows.

Probe gain is a relation-evidence heuristic. It is not the realized marginal value of a newly created rank-2 edge. Post-edit incremental value and learning delay must be logged separately.

The manager may access only the restricted pair-query service for a relation it generated first. It never receives audit-private pair scores or an all-pairs absent-edge table.

## 7. Task 6 amendment — typed evidence, reopening policy, and authoritative gate

Evidence types have different consumers:

```text
CREATE         -> ProbeEvidence for exact proposed endpoints
MODIFY/DORMANT -> ExistingEdgeEvidence for exact edge generation
REOPEN         -> ReopenEvidence for exact VALID handle generation+dormancy version
```

Evidence expires after one structural window unless its constructor declares a shorter lifetime. Successful mutation consumes the evidence reference. Rejected proposals remain provenance without graph mutation.

The gate receives authoritative read-only edge, handle, evidence, proposal, budget, cooldown, rank, and lifecycle views. It computes operation charges from frozen config and current state. Proposal estimates are not authoritative.

The gate rejects stale generations, consumed/revoked handles, wrong dormancy versions, exhausted rank bounds, stale/unknown/consumed evidence, evidence-type mismatches, endpoint mismatches, already-applied proposal IDs, and duplicate active endpoint generations.

The gate never imports world/audit truth or decides whether an edit is scientifically correct.

### Default historical REOPEN policy

An historical handle is eligible for the default reopen scorer only when:

```text
handle state == VALID
historical_support > 0
current endpoint/revision mismatch pressure is present
budget and ordinary gate legality permit reopening
```

`historical_support` is the latest strictly positive leave-one-out contribution from the active epoch immediately preceding that handle's dormancy. It means only that the route was useful then.

The default score may combine:

```text
current pressure
+ historical_support
- authoritative reopening liability
```

with deterministic tie-breaking fixed before execution. It may not use hidden regime information or future loss.

`historical_support = NONE` supplies no default historical reopen path. If a later Bob generation wants to reopen such a route using fresh relation evidence, that is a separately typed design extension rather than an implicit fallback.

## 8. Task 7 amendment — within-step ordering and contribution identity

Each ordinary step must preserve one comparison snapshot:

```text
1. observe x_t
2. encode latents
3. route active messages
4. compute local/full predictions from the same pre-update state
5. reveal x_{t+1}
6. compute L_full and L_local
7. under no_grad, compute each existing-edge leave-one-out contribution from the SAME pre-update parameters, latents, messages, and target
8. update active-epoch latest-positive records only when contribution > 0
9. optimize theta and active phi
10. append detached pre-update probe-buffer records with timestamp/representation version
11. update manager pressure/history
12. at a structural boundary: candidate -> optional paid probe -> typed evidence -> proposal -> gate -> transaction/rejection
13. record observables
14. continue without resetting graph/lineage/manager
```

Existing-edge contribution is:

```math
c(E)=L_{full}(E\text{ masked})-L_{full}(E\text{ active}).
```

The masked and active losses must be computed from the same snapshot. Optimizing between those two measurements is forbidden.

Conformance steps use a fixed W1-only configuration and may inject synthetic proposals to validate accepted mutation→optimizer→ledger→recorder plumbing. They do not run the W1→W2→W3 lifetime.

## 9. Task 8 amendment — route-level corrective-access telemetry

A consequential commitment is admitted by the plan's frozen usage/contribution rule. Once admitted, the **denominator unit is a correction route**, not a commitment.

Freeze route identity as at least:

```text
(commitment_id, route_id, operation, edge_generation, target_rank_or_handle)
```

The then-known V0-semantic correction routes for that consequential commitment enter persistent route cohort `R`. Their route IDs remain in the denominator after dormancy, retirement, handle consumption, or revocation.

Report only:

```math
C_{retained}=\frac{|\{r\in R: retained(r)\}|}{|R|},
```

```math
C_{reachable}=\frac{|\{r\in R: reachable(r)\}|}{|R|}.
```

`retained(r)` means the state/information for that exact route still exists and is not consumed/revoked. `reachable(r)` additionally requires present generation/lifecycle validity, remaining budget, cooldown, and non-oracular gate admissibility.

If `|R|=0`, return `NA` for both fractions and log `route_count=0`. Never return `1.0` because the cohort is empty.

Remove `effective_use` as an access fraction. Demonstrated correction belongs in a separate `CorrectionOutcome` ledger recording route ID, later mismatch/revision episode, exercise time, post-edit evaluation window, task-loss delta, and sustained positive contribution. Initial CREATE and the same window that first makes a commitment consequential cannot count as demonstrated correction.

Even `C_retained=C_reachable=1` does not prove that one bounded policy can identify and choose the correct route across Bob's uncertainty. Keep jointly executable correction as a future Glass Box/MATRIX object.

## 10. Task 9/10 amendment — fail-closed execution identity

Implementation may terminate at:

```text
design_state         = FROZEN
implementation_state = IMPLEMENTED_NOT_EXECUTED
execution_state      = UNEXECUTED
scientific_result    = NONE
```

Any future lifetime runner must verify, inside the runner before world construction or other side effects:

```text
explicit execution authorization
exact design authority commit = deb0fd205b395684531720cdd0a2cd16b9f10ae0
exact separately authorized terminal implementation commit
clean checked-out implementation state
```

The CLI must perform the same checks, but CLI checks do not replace runner checks. Direct Python invocation without valid authorization must refuse.

The implementation commit cannot be self-embedded before it exists; after the terminal implementation commit is created, its exact SHA is recorded externally in the authorization object used for a later run.

Required negative tests include missing authorization, wrong design commit, wrong implementation commit, dirty tracked source, untracked executable source, and direct runner invocation without authorization. The accepted path may dispatch only to a stub/non-scientific runner during implementation validation.

## 11. Claim ceiling

Implementation success would establish only that the repaired Bob Mk I architecture was instantiated and its contracts/tests pass. It would not be a scientific Bob result.

A later observed lifetime could at most support bounded descriptive claims about a neural system instantiating/revising persistent interfaces under this fixed endpoint grammar and cost model. Probe gain remains heuristic; finite-pair search remains finite-pair search; route availability remains weaker than a jointly executable corrective policy; whole-system recovery does not identify a causal mechanism.

## 12. Terminal plan state

This amendment closes the Bob Mk I design/plan repair loop. Task 1 may begin only under a later explicit implementation instruction against this exact documentation lineage. Until then:

```text
Bob Mk I design       FROZEN
Bob Mk I plan         FROZEN
implementation        NOT STARTED
scientific execution  UNAUTHORIZED / UNEXECUTED
scientific result     NONE
```

Any Bob Mk II design must branch from the terminal commit containing this amendment and the repaired design authority, and must explicitly distinguish inherited Mk I contracts from new inquiry-valuation machinery.