# Bob V0 — Revisable Computational Topology Design

**Repository:** `bjoern-janson/revisable-computational-topology`

**Program:** Revisable Computational Topology (RCT)

**Exploratory track name:** `Bob`

**Status:** design approved in conversation; written-spec review pending. This document does not authorize implementation or scientific execution by itself.

## 0. Program boundary

RCT is an experimental architecture program. It is not a canonical extension of Interface Theory, MATRIX, SSI, OpenCore, or any other upstream research program.

Upstream programs may supply source results, constraints, negative results, conceptual pressure tests, or research genealogy. They do not automatically supply evidential support for Bob.

The central RCT hypothesis is prospective:

```math
\boxed{\text{computational topology itself may be a learned, revisable state variable of an adaptive system.}}
```

Bob is the exploratory neural track. A separate Glass Box track will later construct minimal causal assays for mechanisms suggested by Bob.

The division of labor is:

```text
Bob observation
    -> candidate mechanism / hypothesis
    -> Glass Box discriminator
    -> bounded causal result or null
```

Therefore:

```math
\boxed{\text{shared ontology} \neq \text{shared evidence}.}
```

A striking Bob trajectory is an observation, not a mechanism claim.

---

# 1. Bob V0 architectural boundary

## 1.1 One continuing lifetime

Bob V0 operates as one continuing lifetime. There is no episodic reset of computational topology, structural lineage, or topology-manager state.

```math
\mathcal G_t \xrightarrow{e_t} \mathcal G_{t+1} \xrightarrow{e_{t+1}} \mathcal G_{t+2} \rightarrow \cdots
```

The current graph is both the product of prior adaptation and part of the machinery producing future adaptation.

The lifetime is intentionally designed to permit the sequence:

```text
useful structure
-> optimization
-> entrenchment
-> regime shift
-> mismatch
-> reopening / reorganization opportunity
-> new structure
```

The core question is not merely whether Bob recovers reward. It is whether Bob can revise a previously useful structural commitment when later reality makes that commitment wrong.

## 1.2 Core organism state

Define Bob's continuing state as:

```math
\boxed{X_t=(\theta_t,\mathcal G_t,\Lambda_t,\sigma_t)}
```

where:

- `\theta_t` — continuously learned neural parameters;
- `\mathcal G_t` — persistent adapter/interface topology;
- `\Lambda_t` — structural lineage and reopening substrate;
- `\sigma_t` — topology-manager state, including bounded mismatch history, edit pressure, costs, cooldown state, and recent proposal history.

The first causal boundary is:

```math
\boxed{\theta\text{-plasticity} \neq \phi\text{-plasticity} \neq \mathcal G\text{-plasticity} \neq \text{reopening}.}
```

These channels must remain separately observable.

## 1.3 Fixed-topology neural backbone

The backbone is neural and learnable during ordinary operation, but its computational topology is fixed for Bob V0.

For primitive module views:

```math
h_i=f_{\theta,i}(x_t).
```

The parameters may learn:

```math
\theta_t\rightarrow\theta_{t+1},
```

while:

```math
\operatorname{Topology}(f_\theta)=\text{fixed}.
```

The purpose is to preserve an attribution seam between ordinary representation learning and structural topology change.

Bob V0 does not dynamically create arbitrary module semantics.

## 1.4 Mutable adapter graph

Above the fixed-topology backbone lives the mutable interface graph:

```math
\mathcal G_t=(V,E_t).
```

The endpoint vocabulary `V` is fixed in Bob V0. The edge set `E_t` is structurally plastic.

A directed learned interface is a real parameterized neural object:

```math
I_{i\rightarrow j}^{\phi}:H_i\rightarrow H_j.
```

It may maintain:

- interface parameters `\phi`;
- age;
- usage statistics;
- maintenance cost;
- lineage reference;
- structural standing (`active`, `dormant`, etc.);
- reopening information where required.

Directionality is explicit:

```math
I_{ij}\neq I_{ji}
```

in general.

## 1.5 Structural edit vocabulary

Bob V0 supports exactly:

```math
\boxed{\{\operatorname{CREATE},\operatorname{MODIFY},\operatorname{MERGE},\operatorname{DORMANT},\operatorname{RETIRE},\operatorname{REOPEN}\}.}
```

`SPLIT` is intentionally excluded from V0.

The node vocabulary remains fixed. `MERGE` therefore means interface/path consolidation, not module fusion.

`CREATE(i -> j)` means:

> instantiate a new learned interface between already-existing endpoints.

It does not mean:

> invent an arbitrary neural module or arbitrary computational ontology.

Thus Bob V0's first construction claim, if observed cleanly, is about topology generation rather than arbitrary neural-program generation.

## 1.6 Two adaptation clocks

Ordinary parameter adaptation is fast relative to structural edits.

Schematically:

```math
\theta_{t+1}=\theta_t+\eta_\theta\Delta\theta_t
```

and interface-parameter adaptation may update:

```math
\phi_{ij,t}\rightarrow\phi_{ij,t+1}.
```

Structural topology changes occur on a slower event process:

```math
\mathcal G_{t+1}=
\begin{cases}
\operatorname{Edit}(\mathcal G_t,a_t), & \text{if the structural gate admits an edit},\\
\mathcal G_t, & \text{otherwise}.
\end{cases}
```

The exact numerical pressure formula is not part of this design specification. It is an implementation choice unless later frozen in a Glass Box assay.

The design requirement is causal ordering:

```text
prediction error
-> ordinary theta adaptation first
-> interface-parameter adaptation opportunity
-> persistent structured mismatch
-> structural edit becomes eligible
```

A topology change should therefore be an earned slow response to persistent structural pressure rather than a noisy response to one loss spike.

## 1.7 Structural transactions and lineage

Every accepted graph mutation is recorded as a first-class structural transaction:

```math
T_k=(G^-,a_k,G^+,b_k,D_k,H_k,c_k)
```

where:

- `G^-` — topology before the edit;
- `a_k` — accepted structural edit;
- `G^+` — topology after the edit;
- `b_k` — evidence / mismatch basis available at decision time;
- `D_k` — affected or discarded structural routes/distinctions, if any;
- `H_k` — reopening handle;
- `c_k` — creation, maintenance, and reopening liabilities.

Then:

```math
\Lambda_t=\{T_1,\ldots,T_n\}.
```

The transaction record is immutable historical provenance. Later success or failure must not rewrite the basis on which an earlier edit was made.

## 1.8 Reopening liability

For Bob V0, use:

```math
\boxed{\rho(T)=\big(\rho_{\rm info},\rho_{\rm reach},\rho_{\rm effective-use}\big).}
```

The components mean approximately:

- `rho_info` — cost of recovering enough retained structural information to reconstruct the route;
- `rho_reach` — cost of reaching the route / reconstruction process from the current state;
- `rho_effective-use` — cost of making the restored route operationally usable again.

This terminology is intentionally architecture-local and does not import broader authority semantics from upstream programs.

## 1.9 Narrow reopening semantics

For V0:

```math
\boxed{\operatorname{REOPEN}(T_k)\neq\operatorname{RESTORE\_CHECKPOINT}(t-k).}
```

Reopening means:

> restore an explicitly retained alternative structural route made unavailable by a prior transaction, subject to its recorded reopening costs.

The lifetime continues. The restored route may be modified again immediately.

---

# 2. Continuing neural world — Rotating Dependency World V0

## 2.1 Purpose

The world must create the following pressure:

```math
\boxed{\text{local representations remain healthy} \quad\land\quad \text{learned relationships become wrong}.}
```

This distinguishes a relational/topological regime change from a trivial local distribution shift.

## 2.2 Hidden latent system

Use four persistent neural sensor domains:

```math
A,B,C,D.
```

Each domain observes a nonlinear noisy view of a hidden latent block:

```math
z_t=(z_A,z_B,z_C,z_D)
```

with each `z_i in R^d`, and:

```math
x_i(t)=\psi_i(z_i(t))+\xi_i.
```

The unknown fixed nonlinear emissions `\psi_i` ensure that the backbone must learn useful local latent representations rather than receiving hand-coded state bits.

## 2.3 Regime-dependent dependency topology

Use a latent transition family of the form:

```math
z_{t+1}=\rho P_rQz_t+\sqrt{1-\rho^2}\,\epsilon_t,
```

where:

- block dynamics are identically distributed;
- innovations are isotropic;
- `Q` contains fixed unknown orthogonal / within-block transformations;
- `P_r` permutes whole blocks according to the hidden regime.

The exact implementation may use an equivalent construction, but it must make marginal invariance a designed property rather than a hoped-for empirical side effect.

## 2.4 Explicit marginal-invariance contract

The environment must satisfy, by construction and audit:

```math
\boxed{p(x_i\mid W_1)=p(x_i\mid W_2)}
```

up to declared numerical tolerance for every individual sensor stream `i`.

What changes is relational structure, e.g.:

```math
p(x_j(t+1)\mid x_i(t),W_1)\neq p(x_j(t+1)\mid x_i(t),W_2).
```

A predeclared environment diagnostic, independent of Bob's training objective, must verify that the intended relational shift occurred without detectable marginal drift beyond tolerance.

## 2.5 Lifetime regimes

Bob experiences one continuous trajectory with three environmental phases.

### W1 — Formation

The hidden dependency graph rewards approximately:

```text
A <-> B
C <-> D
```

Useful learned interfaces should have an economic reason to emerge, but no specific graph trajectory is required.

### W2 — Entrenchment

The hidden regime remains unchanged.

The purpose is to let useful learned topology mature into infrastructure: interface parameters specialize, usage accumulates, maintenance economics matter, and redundant routes may become unattractive.

Entrenchment is not defined merely by age; Section 4 defines observable dependence.

### W3 — Structural inversion

The hidden dependency topology rotates to approximately:

```text
A <-> C
B <-> D
```

while preserving each local marginal law.

Previously useful relations become systematically misleading. The intended challenge is that local sensor representations can remain well calibrated while relational predictions tied to old topology deteriorate.

The manager never receives `W1`, `W2`, `W3`, the hidden permutation, or the correct new edge identities.

## 2.6 Distinguishing local error from topology error

Track separate classes of local and relational losses.

The intended post-shift signature is:

```math
L_{\rm local}\approx\text{stable}
\qquad\land\qquad
L_{\rm relational,old}\uparrow.
```

Ordinary `theta` and existing-interface `phi` updates must have an opportunity to adapt before structural change becomes eligible.

Thus the intended causal sequence is:

```text
world changes
-> local calibration remains approximately healthy
-> existing relation residual rises
-> theta adaptation opportunity
-> phi adaptation opportunity
-> structured residual persists
-> topology edit pressure accumulates
```

This sequence provides an attribution surface; it does not predetermine which structural edit is correct.

## 2.7 Structural economics

Bob's lifetime objective must penalize unnecessary topology.

A schematic objective is:

```math
J=-\sum_t L_{\rm task}(t)-\lambda_M C_{\rm maintenance}(\mathcal G_t)-\lambda_E C_{\rm edit}(a_t)-\lambda_R C_{\rm reopen}(a_t).
```

The exact coefficients are implementation parameters until separately frozen.

Required qualitative economics:

- `CREATE` has one-time construction cost plus maintenance cost;
- `MODIFY` is cheaper than creation but may fail to remove persistent mismatch;
- `DORMANT` reduces maintenance while preserving a relatively cheap reopening path;
- `RETIRE` saves more but increases future reconstruction/recovery liability;
- `REOPEN` pays recorded reopening liabilities.

No positive reward is assigned merely for graph complexity or edit count.

## 2.8 No trajectory scripting

The environment freezes latent laws and structural costs, not Bob's path.

The design does **not** predeclare that Bob must:

- create `A -> C`;
- dormancy `A -> B`;
- reopen a particular historical transaction;
- use every structural primitive.

Unexpected effective topology is explicitly allowed and is a primary reason Bob exists as an exploratory track.

---

# 3. Topology manager — observation, proposal, and construction constraints

## 3.1 Central construction boundary

Freeze:

```math
\boxed{\mathcal G\text{-generation}\neq\text{scoring a designer-supplied candidate universe}.}
```

And freeze the key rule:

```math
\boxed{\textbf{The manager may purchase information about a proposed relation; it may not receive precomputed information about unproposed relations.}}
```

The manager must not receive an all-pairs table that implicitly solves topology generation for it.

## 3.2 Manager observation interface

The manager is a persistent, regime-blind process:

```math
\mathfrak G_t:(O_t^{\rm mgr},\sigma_t,\Lambda_t)\rightarrow a_t.
```

Its ordinary dashboard may contain:

```math
O_t^{\rm mgr}=(S_t^M,S_t^E,C_t,\Lambda_t,H_t),
```

where:

- `S_t^M` — per-module calibration, uncertainty, activity, and other bounded local summaries;
- `S_t^E` — statistics for currently existing interfaces only;
- `C_t` — current structural costs;
- `H_t` — bounded recent mismatch, proposal, and edit history.

The manager may observe consequences of current structure.

It must never receive:

- hidden regime identity;
- hidden dependency matrix;
- hidden block permutation;
- correct-edge labels;
- precomputed loss/ranking for all absent edges;
- privileged future loss;
- counterfactual world rollouts that reveal the correct structural edit.

## 3.3 Existing-edge evidence versus absent-edge evaluation

For current edges, Bob may maintain ordinary relational residual statistics.

For absent edges, the manager may not continuously receive `L_relational,ij` for all pairs.

Freeze:

```math
\boxed{\text{existing-edge evidence}\neq\text{absent-edge evaluation}.}
```

Otherwise candidate generation degenerates into argmax/argmin over a designer-computed table.

## 3.4 Candidate relation generation

The manager first generates a candidate directed relation from its current information:

```math
\mathfrak G_{\rm candidate}(O_t^{\rm mgr},\sigma_t,\Lambda_t)\rightarrow(i,j).
```

This step chooses what relationship is worth investigating before Bob knows whether that relation is useful.

No implementation strategy is frozen here. The manager may later be implemented as a learned policy, recurrent controller, hybrid search process, or another bounded mechanism, provided it respects the observation constraints.

## 3.5 RELATION_PROBE

Introduce a diagnostic action that is **not** a graph edit:

```math
\boxed{\operatorname{RELATION\_PROBE}(i\rightarrow j).}
```

It temporarily asks whether currently available latent information at endpoint `i` appears useful for reducing unexplained relational residual at endpoint `j`.

The probe is:

- temporary;
- read-only;
- costly;
- rate-limited / budgeted;
- nonpersistent as a topology object;
- unable to rewrite modules;
- unable to create an interface directly.

Therefore:

```math
\operatorname{RELATION\_PROBE}\notin\{CREATE,MODIFY,MERGE,DORMANT,RETIRE,REOPEN\}.
```

## 3.6 Probe budget is an accounted lifetime resource

For each slow structural window `w`, define a probe budget:

```math
B_{\rm probe}(w).
```

Every relation probe consumes documented cost, and:

```math
\boxed{N_{\rm probe}(w)\ll |V|(|V|-1).}
```

Bob is therefore unable to exhaustively test all directed candidate relations before committing.

The exact ratio is implementation-specific until later frozen, but exhaustive all-pairs evaluation is forbidden.

## 3.7 Probe evidence is not structural mutation

A probe produces evidence:

```math
\operatorname{RELATION\_PROBE}(i,j)\rightarrow e_{ij}.
```

It does not directly create an edge.

There must remain an explicit decision chain:

```math
e_{ij}\rightarrow P_k\rightarrow\operatorname{Gate}\rightarrow T_k.
```

Freeze:

```math
\boxed{\text{evidence}\neq\text{decision}\neq\text{structural mutation}.}
```

## 3.8 Proposal record

Every proposed structural edit is represented before acceptance as:

```math
P_k=(O_k,a_k,\widehat{\Delta V}_k,\widehat C_k,R_k),
```

where:

- `O_k` — manager-visible evidence actually available at proposal time;
- `a_k` — candidate edit;
- `\widehat{\Delta V}_k` — estimated benefit;
- `\widehat C_k` — estimated structural cost;
- `R_k` — routes / structural objects expected to be affected.

Rejected proposals remain part of provenance.

An accepted proposal creates a structural transaction:

```math
P_k\rightarrow T_k.
```

This permits later analysis of what Bob expected when a structural commitment was made.

## 3.9 CREATE semantics

`CREATE(i -> j)` creates a previously absent parameterized interface:

```math
I_{i\rightarrow j}^{\phi}:H_i\rightarrow H_j.
```

If bidirectional communication is useful, the reverse edge must be separately proposed and constructed.

For Bob V0:

```math
\boxed{\text{CREATE generates topology, not arbitrary interface semantics.}}
```

The manager generates the endpoint relation and may optionally provide a bounded initialization code, but the interface weights `\phi` learn through ordinary neural optimization after creation.

Therefore Bob V0 may earn:

> a previously absent neural interface was instantiated from a system-generated endpoint proposal.

It does not earn:

> arbitrary neural program synthesis.

## 3.10 Structural gate

Separate proposal from permission to mutate the graph.

The flow is:

```text
current consequences
-> structural pressure
-> candidate generation
-> optional paid relation probe
-> proposal P_k
-> structural gate
-> accepted transaction T_k or rejection
```

The gate enforces architecture-level validity only, such as:

- sufficient accumulated mismatch under the configured policy;
- cooldown/resource constraints;
- legal endpoints;
- legal operation type;
- lineage/reopening record constructibility;
- forbidden-oracle-field absence;
- probe-budget accounting.

The gate must not know which edit is scientifically correct.

Freeze:

```math
\boxed{\text{admissible edit}\neq\text{good edit}.}
```

Bob must be allowed to make bad structural choices.

## 3.11 REOPEN proposal semantics

Reopening candidates come from current evidence plus structural lineage:

```math
\mathfrak G_{\rm reopen}(O_t^{\rm mgr},\Lambda_t)\rightarrow T_k.
```

The world never supplies the identity of the transaction that should be reopened.

The manager must infer whether a historical commitment is implicated by current mismatch.

If admitted:

```math
\operatorname{REOPEN}(T_k)
```

pays the recorded reopening liability and restores the explicitly retained structural alternative according to Section 1.

## 3.12 Frozen anti-cheating exclusions

Bob V0 forbids:

```text
true regime ID
true dependency graph
changed permutation
correct-edge labels
all-pairs absent-edge scoring
exhaustive edge trial before commitment
privileged future loss
privileged counterfactual world rollouts
post-hoc graph selection presented as online adaptation
arbitrary module creation
arbitrary interface-program synthesis
probe result -> automatic CREATE without proposal/gate
```

---

# 4. Bob V0 observability contract

## 4.1 Purpose

Bob is an exploratory research organism, not a conventional benchmark whose desired structural trajectory is predeclared.

Section 4 therefore defines observable classes and records rather than a single success score.

The principle is:

```math
\boxed{\text{observe trajectories without scripting trajectories}.}
```

## 4.2 Structural trajectory record

Bob must emit a continuing structural trajectory sufficient to reconstruct its life:

```math
\mathcal T=\{(\mathcal G_t,\Lambda_t,\sigma_t,V_t,C_t)\}_{t=1}^{T}.
```

The exact storage frequency may be reduced through event logging plus periodic snapshots, but the resulting record must permit reconstruction of graph state, structural edits, proposal history, probe expenditure, and measured task/corrective proxies over time.

## 4.3 Five core observables

Bob V0 records at least:

```text
1. performance / viability trajectory
2. topology trajectory
3. structural transaction lineage
4. probe / exploration expenditure
5. reopening / recovery trajectory
```

These are descriptive observables. None by itself establishes a mechanism claim.

## 4.4 Descriptive structural phases

The analysis may classify periods using labels such as:

```text
FORMATION
ENTRENCHMENT
MISMATCH
REORGANIZATION
RECOVERY
```

These labels describe what occurred. They do not prescribe the graph edit sequence that should occur.

## 4.5 Operational structural adaptation

A structural edit may be called operationally useful when it produces a measurable improvement that persists beyond the immediate edit/transient window after accounting for its declared costs.

No particular edit type is required.

Useful adaptation may arise through:

- `CREATE`;
- `MODIFY`;
- `DORMANT`;
- `REOPEN`;
- `MERGE`;
- `RETIRE`;
- or a sequence of legal edits.

The mechanism producing any observed improvement remains unearned until separately isolated.

## 4.6 Entrenchment

For an interface `I`, record an entrenchment descriptor such as:

```math
E(I)=(\text{usage},\text{performance contribution},\text{maintenance cost},\text{age},\text{dependence count}).
```

Exact operational thresholds may be chosen during implementation and must be reported.

Conceptually, an interface is entrenched when the organism has become materially dependent on it under the current regime.

Age alone is insufficient.

## 4.7 Functional reopening

Do not define reopening by similarity to a historical checkpoint.

A reopening event is functionally:

```text
current structural state
-> historically implicated commitment
-> previously unavailable structural route restored
```

The restored route may immediately be transformed again.

Thus:

```math
\boxed{REOPEN\neq RESTORE\_CHECKPOINT.}
```

## 4.8 Performance and corrective-capacity trajectories

Track at least two conceptually separate quantities:

```math
V_t=\text{task / viability performance proxy}
```

and a Bob-local corrective-access proxy:

```math
C_t=\text{measured reachable capacity to revise currently consequential structural commitments}.
```

`C_t` is not imported as a universal corrigibility scalar. Its exact operationalization belongs to the implementation/measurement design and must remain explicitly scoped to Bob V0.

Interesting descriptive regimes include:

```math
V_t\uparrow,\quad C_t\uparrow
```

and the danger-shaped trajectory:

```math
V_t\uparrow,\quad C_t\downarrow.
```

A particularly interesting observation would be:

```text
V rises
-> C falls
-> regime shift
-> V falls
-> reopening / reorganization
-> C rises
-> V recovers
```

Such a trajectory would be scientifically interesting but would not, by itself, establish a general law or a specific causal mechanism.

## 4.9 Bob observation versus Glass Box evidence

Keep three levels separate.

**Bob observation**

Example:

> Bob created `I_AC`, later dormanted `I_AB`, then reopened transaction `T_7` before recovery.

**Glass Box hypothesis**

Example:

> Access to the structural route reopened by `T_7` was causally necessary for recovery after the regime shift.

**Glass Box experiment**

Example:

> Compare matched conditions with reopening access enabled versus blocked while holding other relevant pathways fixed.

Therefore:

```math
\boxed{\text{Bob generates phenomena}\rightarrow\text{Glass Box adjudicates mechanisms}.}
```

## 4.10 No predeclared expected graph trajectory

Bob V0 must not define success as a required sequence such as:

```text
CREATE A->C
DORMANT A->B
REOPEN T_k
```

Unexpected but effective graph organization is a first-class outcome.

The scientifically valuable result may be that Bob discovers a structural strategy the designers did not expect.

The correct next action is then to formulate a discriminating Glass Box question, not to reinterpret the observation as proof of a broad architectural theory.

---

# 5. Bob / Glass Box relationship

Bob and Glass Box share primitives where useful, but not evidential standing.

Candidate shared primitives include:

```text
Module
Interface
GraphEdit
Evidence record
Proposal
Transaction
Lineage
ReopenHandle
EnvironmentEvent
```

Bob is allowed to be exploratory, surprising, inefficient, and wrong.

Glass Box is allowed to be tiny, boring, and decisive.

The research loop is:

```text
Bob
-> observable structural phenomenon
-> candidate explanation
-> Glass Box minimal causal discriminator
-> result / null / localized failure
-> updated Bob question or architecture
```

Component success does not establish composition success, and Bob's whole-system success does not retrospectively identify every component mechanism.

---

# 6. Bob V0 non-claims

Bob V0 does not claim in advance that:

- dynamic topology is superior to a conventional fixed neural architecture;
- topology revision will occur;
- reopening will occur;
- reopening will improve recovery;
- the manager will generate genuinely novel scientific concepts;
- topology generation is equivalent to representation generation;
- a good Bob trajectory establishes safe self-improvement;
- `C_t` is a universal corrigibility measure;
- Interface Theory proves Bob's architecture;
- MATRIX proves Bob's architecture;
- upstream formal results translate directly into Bob mechanisms;
- any one successful lifetime establishes generality;
- a learned edge implies the claimed reason for its creation;
- task recovery implies structural reopening was necessary;
- structural adaptation implies beneficial long-horizon improvement.

---

# 7. Minimal implementation target after written-spec approval

The first implementation should be deliberately small.

It should contain only enough machinery to instantiate the frozen design boundary:

```text
neural fixed-topology backbone
persistent directed adapter graph
one continuing Rotating Dependency World lifetime
basic topology manager observation interface
candidate relation generation
budgeted RELATION_PROBE
CREATE / MODIFY / DORMANT / REOPEN path
structural proposal + gate + transaction logging
trajectory recorder
basic visualization / inspection surface
```

`MERGE` and `RETIRE` may be represented in the core edit vocabulary but do not need to be forced into the first behavioral demonstration.

No additional abstraction layer should be added before this small version is built and observed unless implementation exposes a concrete missing boundary.

---

# 8. Design stopping rule

This specification intentionally stops here.

The next step after user review is implementation planning, not further architectural expansion.

The purpose of Bob V0 is to create enough genuine neural and structural freedom for the system to surprise us while preserving enough observability to turn surprising trajectories into falsifiable Glass Box questions.

North-star question:

```math
\boxed{\textbf{Can a neural organism discover and revise its own computational organization because reality changed, while preserving enough structural lineage to discover that its old organization was the problem?}}
```
