# Bob Mk II — durable chat handoff

Date: 2026-09-11

Standing: `HANDOFF_FROZEN / LIFETIME_001_EXECUTED / WARRANT_REPAIR_VERIFIED / LIFETIME_002_NOT_AUTHORIZED / LIFETIME_002_NOT_EXECUTED`

This file is the durable handoff for the long ChatGPT research session that reached its conversation-length limit. It is intentionally explicit about lineage, claim ceilings, negative results, and the next legal action. It is not itself an execution authorization.

## 1. Repository and authoritative lineage

Repository: `bjoern-janson/revisable-computational-topology`

Authoritative lineage relevant to Bob Mk II:

```text
7996daff3269da8e6e383cccd305eee990fead1a
  original Bob Mk I docs base
    ↓
deb0fd205b395684531720cdd0a2cd16b9f10ae0
  Bob Mk I repaired DESIGN authority
    ↓
37f8b3df554ddffcfdf4b17a51b3fd703294ae78
  Bob Mk I repaired PLAN terminal
    ↓
716118147c5ec8925eaa1c3ddc3b5f500eb51c29
  Bob Mk II approved design
    ↓
2259c818...   Bob Mk II implementation plan
    ↓
38cdaebdc9ad6dec3fc1a4d4c9b279a73ad1c0a6
  plan self-review / terminal plan state
    ↓
095505276faf0faaad02d71241f1d881999396f8
  terminal implementation executed in Lifetime 001
    ↓
660774a0154dc184d1896ee76dc4a441d18cae23
  frozen Lifetime-001 result record
    ↓
e3be1166ac0a6e7d4abde7856a6d9d53719410fb
  narrow probe-warrant continuity repair
    ↓
6b9e58356aca1bc0e5f014ab5c4d33878bc08a05
  engineering prefix-replay result record
```

The plan's intermediate commit `2259c818...` should be resolved from Git history if its full SHA is needed; the authoritative terminal plan SHA is `38cdaebdc9ad6dec3fc1a4d4c9b279a73ad1c0a6`.

Important branches:

- `result/bob-mk-ii-lifetime-001` — frozen Lifetime-001 result, head `660774a0154dc184d1896ee76dc4a441d18cae23`.
- `repair/bob-mk-ii-probe-warrant-continuity` — repaired implementation, head `e3be1166ac0a6e7d4abde7856a6d9d53719410fb`.
- `result/bob-mk-ii-warrant-repair-replay` — engineering replay record, head `6b9e58356aca1bc0e5f014ab5c4d33878bc08a05`.
- `archive/bob-mk-ii-handoff-2026-09-11` — this durable handoff/archive branch. It is for custody, not new science.

Do not rewrite the frozen result branches to make later interpretation look prospective.

## 2. Global scientific discipline

Preserve these distinctions:

```text
event != observation != evidence != warrant != authority != persistence != truth
internal consistency != formal soundness != empirical validity != generality
component success != composition success
valid state != valid transition != valid composition
stored != recoverable != reopenable != actionable != causally effective
```

A null or failure at layer k localizes the failure; it does not collapse the whole architecture.

For Bob Mk II specifically:

```text
evidence != candidate != learned valuation != proposal != gate permission != mutation
```

The learned valuation may rank actions. It may never manufacture evidence, warrant, permission, budget, lifecycle validity, or structural authority.

## 3. Bob Mk II architecture frozen before Lifetime 001

Bob's state extends the Mk I structural agent with a persistent learned manager-value state `eta`.

The manager ranks the legal action vocabulary:

```text
PROBE
CREATE
MODIFY_UP
MODIFY_DOWN
DORMANT
REOPEN
RETIRE
ABSTAIN
```

The value model is one shared inspectable linear model:

```math
Q_eta(s,a) = eta^T phi(s,a)
```

with 38 prospectively fixed coordinates grouped around current pressure/contribution, resource cost, future-access structure, lineage/history, and action identity. The exact feature formulas live in the frozen design/plan and implementation; do not silently redesign them during interpretation.

Critical seam:

```text
phi_immediate != phi_future_access
```

This preserves a future Glass Box ablation target.

The future-access features use only Bob-visible typed metadata/current represented options. They must not use the hidden true graph, future task loss, oracle-correct relation labels, or audit-private world pairings.

### Training target

The manager is trained on completed realized finite-horizon return, not TD bootstrapping:

```math
Y_t = sum_{tau=t}^{t+255} [
    -L_task(tau)
    -C_probe(tau)
    -C_edit(tau)
    -C_maintenance(tau)
    -C_reopen(tau)
]
```

The frozen implementation uses `H=256` ordinary steps and `gamma=1`.

`C_retained` and `C_reachable` are observational telemetry only. They are not rewarded. Therefore Bob is not paid directly for satisfying the researcher's corrigibility metric.

The learner is completed-return ridge regression with neutral initialization, not Q-learning. Overlapping 256-step training returns may be used by the learner but are not independent scientific replications.

### Exploration

Reversible action classes receive bounded prospectively fixed exploration. `RETIRE` is never randomly forced. `RETIRE` can enter learned competition only after its separate estimator-support gate says its region is supported. Exploration provenance and greedy provenance are logged separately.

### Structural authority

The structural gate remains authoritative and nonlearned. The manager only sees/ranks currently formable candidates. Structural actions continue to require typed evidence/lifecycle/budget/cooldown legality.

Mk I lifecycle semantics remain in force, including generation-aware edge identity, dormancy-versioned handles, stale-handle rejection, and `contribution > 0` as the active-epoch positive-support condition. Historical support means useful then; it is not present truth.

## 4. Lifetime 001 — exact custody

Frozen result record:

`docs/results/2026-09-10-bob-mk-ii-lifetime-001.md`

Result branch head:

`660774a0154dc184d1896ee76dc4a441d18cae23`

Execution identity:

- Design: `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`
- Implementation: `095505276faf0faaad02d71241f1d881999396f8`
- Authorization: `AUTH-USER-20260910-FIRST-LIFETIME`
- Process exit: `0`
- World schedule, disclosed after completion: `W1=2476`, `W2=2478`, `W3=1632`
- Ordinary steps / raw trajectory rows: `6586`

Raw Lifetime-001 trajectory identity:

- raw JSONL SHA-256: `b955334bc4c1129dca7fbb0ee8f6c3d5246b760f96cad6e2db185bccac95cf7e`
- raw bytes: `10,396,380`
- deterministic `gzip -n -9` SHA-256: `23b201977545e3839bcaddcf58d0f9ba793047348bdcffe78d83876c65460f2e`
- deterministic gzip bytes: `526,423`

A base64-encoded copy of that deterministic gzip is stored on this archive branch at:

`results/archive/bob-mk-ii-lifetime-001/trajectory.jsonl.gz.b64`

Recover with:

```bash
base64 -d trajectory.jsonl.gz.b64 > trajectory.jsonl.gz
gzip -dc trajectory.jsonl.gz > trajectory.jsonl
sha256sum trajectory.jsonl
```

The final raw hash must equal `b955334b...` above.

## 5. Lifetime 001 — observed result

Observed terminal facts:

- manager decisions: `102`
- chosen actions: `100 ABSTAIN`, `2 PROBE`
- structural transactions: `0`
- maximum active edges: `0`
- maximum correction-route count: `0`
- matured valuation decisions at terminal step: `98`
- pending valuation decisions: `4`
- structural fingerprint: constant over the whole lifetime

Thus Bob's ordinary neural parameters learned, but Bob's topology did not change.

The two probes occurred at ordinary steps `575` and `4863`, each costing `0.02`.

At step `575`, the eligible set was:

```text
PROBE:C->D
PROBE:A->D
ABSTAIN
```

The two probe candidates had identical 38-coordinate feature vectors and identical predicted values (`-146.92291422887894`). Deterministic tie-breaking selected `PROBE:A->D`. Audit-private post-run world knowledge says `C-D` was the relevant W1 pairing.

At step `4863`, the eligible set was:

```text
PROBE:A->C
PROBE:B->C
ABSTAIN
```

Again the two probes had identical feature vectors/scores (`-188.85963957269473`). Bob selected `PROBE:B->C`. Audit-private post-run world knowledge says the useful W2 partner for C was D, so the useful relation was not generated at all.

At the immediately following structural boundaries, `639` and `4927`, only `ABSTAIN` was eligible even though the corresponding `ProbeEvidence` remained fresh. The gate re-required the original three qualifying inquiry-pressure windows for CREATE. The detector condition had fallen away.

Therefore the frozen Lifetime-001 localization is:

```text
DETECT/GENERATE -> QUERY occasionally succeeds -> QUERY !-> structural ACT
```

and, more specifically:

```text
qualified pressure -> PROBE -> fresh evidence -> pressure qualification disappears -> CREATE blocked
```

Lifetime 001 is an upstream generation/detection/warrant-composition failure. It is **not** evidence against learned future-corrective-access valuation, because no structural commitment ever existed whose correction routes could later matter.

### Lifetime-001 task-loss telemetry

Mean full task loss:

- W1: `0.7789260341224647`
- W2: `0.7287719342331428`
- W3: `0.7206823990917673`

Transition diagnostic only:

- trailing 256-step W2 mean: `0.7042340680491179`
- first 256-step W3 mean: `0.7168430557940155`

These are descriptive observations, not causal estimates of structural adaptation.

Delayed-return observations for the two probes:

- D000008 / step 575: prediction `-146.92291422887894`, realized 256-step return `-199.8969029676914`, residual `-52.973988738812466`.
- D000075 / step 4863: prediction `-188.85963957269473`, realized 256-step return `-178.2621405315399`, residual `10.597499041154833`.

These are realized associations under the continuing policy, not causal effects of probing.

## 6. Narrow repair prospectively approved after Lifetime 001

The user approved one bounded repair only: preserve a legally acquired probe's **acquisition warrant** long enough for fresh evidence to be acted upon at the next structural boundary.

The repair intentionally does **not** change candidate generation, probe aliasing, the world, eta, features, costs, exploration, action vocabulary, or one-action-per-boundary sequencing.

Repaired causal/type path:

```text
qualified PROBE
  -> authoritative gate issues typed ProbeAcquisitionWarrant
  -> probe service acquires ProbeEvidence(warrant_ref)
  -> next boundary CREATE may use matching fresh unconsumed evidence+warrant
  -> successful CREATE consumes both evidence and warrant
```

Key principle:

```text
ProbeEvidence != ProbeAcquisitionWarrant != gate permission
```

`ProbeEvidence` carries only a reference to the warrant. The authoritative gate issues the warrant only after PROBE passes its existing formability/qualification rules. Missing, stale, consumed, or endpoint-mismatched evidence/warrant must fail closed.

CREATE no longer re-tests the original three-window inquiry-pressure condition when a valid fresh acquisition warrant and its matching fresh evidence exist. This exception is only for CREATE's handoff; it is not sticky pressure and does not change other operation gates.

## 7. Repaired implementation exact state

Authoritative repair branch:

`repair/bob-mk-ii-probe-warrant-continuity`

Repair implementation SHA:

`e3be1166ac0a6e7d4abde7856a6d9d53719410fb`

It is exactly one commit ahead of Lifetime-001 result record `660774a...`, with zero commits behind.

Files changed by the repair relative to `660774a...`:

```text
src/rct/bob/gate.py
src/rct/bob/lifetime.py
src/rct/bob/probe.py
src/rct/bob/types.py
src/rct/bob/warrant.py          [new]
tests/test_gate.py
tests/test_probe.py
tests/test_probe_warrant_continuity.py [new]
```

No world/config/feature/valuation/manager/exploration file changed.

GitHub Actions checked out exact `e3be1166...` and reported:

```text
86 passed in 13.32s
SOFTWARE_CONFORMANCE_ONLY steps=8 scientific_result=NONE
```

Therefore `e3be1166...` is the current repaired implementation identity eligible for a future fresh scientific lifetime, subject to a new explicit authorization.

## 8. Same-seed engineering prefix replay

Frozen replay record:

`docs/results/2026-09-10-bob-mk-ii-probe-warrant-repair-replay.md`

Result branch:

`result/bob-mk-ii-warrant-repair-replay`

Result record head:

`6b9e58356aca1bc0e5f014ab5c4d33878bc08a05`

Standing:

`ENGINEERING_REPLAY_PASS / NOT_FRESH_SCIENCE / NO_FULL_LIFETIME`

The replay reused the same public configuration/seeds and ran ordinary steps `0..639` only.

Raw replay custody:

- rows: `640`
- raw JSONL SHA-256: `b067eea11a257310332eecd13bf59ef18d0bb74d11ea1e76cdc408fd3a38b0b2`
- raw bytes: `903,530`
- deterministic `gzip -n -9` SHA-256: `708b7c99dadfb85d7954ab8770f9a81cefcb556844b3538d1a88f76b6438dc31`
- deterministic gzip bytes: `50,330`

A base64-encoded copy of the deterministic gzip is stored on this archive branch at:

`results/archive/bob-mk-ii-warrant-repair-prefix-replay/trajectory.jsonl.gz.b64`

The replay reproduced the original step-575 candidate set and choice:

```text
575: PROBE:A->D
     evidence E000001
     acquisition warrant W000001
     validity through step 639 inclusive
```

At step 639, after original pressure qualification had disappeared, the repaired eligible set was:

```text
CREATE:A->D:E000001
ABSTAIN
```

Bob selected CREATE. The gate returned `allowed=True`, `reason=FORMABLE`, transaction `T000001`. Active-edge count became 1. `E000001` and `W000001` were both consumed.

This validates only the repaired handoff under a same-seed engineering replay. The fact that CREATE happened is downstream behavior in that replay, not an independently randomized endpoint.

## 9. Still-open obstruction: probe candidate aliasing / generation

Do not silently claim this was repaired.

Lifetime 001 showed two distinct candidate-generation problems:

1. At step 575, `PROBE:C->D` and `PROBE:A->D` had identical visible feature vectors and scores, even though post-run audit-private knowledge says C-D was the relevant W1 relation. The deterministic tie-break chose A-D.
2. At step 4863, the useful relation D->C was not in the generated probe candidate set at all.

This is upstream of future-access valuation. A future experiment may need to distinguish:

```text
generation failure
vs
representation/feature aliasing
vs
valuation failure
```

but no such repair has been prospectively frozen or implemented yet.

The approved warrant-continuity repair deliberately leaves this rock untouched.

## 10. CorrectionOutcome remains unresolved

The codebase has a `CorrectionOutcome`-style observational concept, but the conversation never prospectively froze an exact post-edit baseline/evaluation-window rule strong enough to generate demonstrated-correction labels without discretion.

Do not backfill such a rule after seeing Lifetime-001 or future outcomes. If demonstrated correction becomes a scientific endpoint, first freeze an exact measurement contract prospectively.

## 11. Current claim ceiling

What is established:

- Bob Mk II implementation `095505...` executed one full lifetime successfully.
- Lifetime 001 made no structural changes and localizes a PROBE-to-CREATE warrant-composition failure plus upstream candidate-generation/aliasing issues.
- The narrow typed acquisition-warrant repair `e3be1166...` removes the specific temporal handoff obstruction in focused tests and the same-seed 640-step engineering replay.
- The repaired implementation has independent GitHub Actions software-conformance verification, `86/86` tests.

What is **not** established:

- that repaired Bob improves task performance;
- that the A->D edge created in the engineering replay is useful;
- that Bob will adapt correctly in W2 or W3 under the repaired implementation;
- that `phi_future` causally affects action selection;
- that feedback teaches Bob to preserve future corrective access;
- safe forgetting;
- MATRIX's safe-quotient frontier in Bob;
- general corrigibility;
- query invention;
- representation invention;
- general topology self-revision;
- recursive self-improvement.

In particular, the main intended Mk II question remains untested:

```text
Can delayed realized task/resource feedback train a compact persistent valuation
that prefers legal inquiry/topology actions partly because of the future corrective
options those actions leave available?
```

## 12. Next legal action

No Lifetime 002 has been authorized or executed.

The clean next scientific action is a **fresh full W1->W2->W3 Lifetime 002** bound to repaired implementation:

`e3be1166ac0a6e7d4abde7856a6d9d53719410fb`

It requires a new explicit user authorization. Suggested authorization phrase:

```text
authorized lifetime 002
```

A new authorization record should be SHA-bound to `e3be1166...`; do not reuse `AUTH-USER-20260910-FIRST-LIFETIME`.

Before Lifetime 002 starts, independently verify:

```text
HEAD == e3be1166ac0a6e7d4abde7856a6d9d53719410fb
tracked executable state clean
no untracked executable files
software conformance tests green
world not constructed before authorization gate
```

Then run the full lifetime once and preserve the raw append-only trajectory before interpretation.

If Lifetime 002 reaches structural actions, do not leap straight to a future-access conclusion. First localize:

```text
DETECT/GENERATE
-> QUERY
-> typed warrant/evidence
-> structural ACT
-> topology state
-> route retention/reachability
-> later corrective opportunity
-> later realized task/resource consequence
```

Only layers actually exercised may receive credit.

## 13. Broader research spine relevant to interpretation

The user's current program spine is:

```math
I proportional to C_improve = V_future(with feedback) - V_future(without feedback)
```

with the causal intuition:

```text
feedback
-> better representation
-> better adaptive mechanisms
-> greater improvement capacity
-> expanded viable futures
```

Improvement is a comparison of future viability with versus without feedback; it is not a behavior label.

The recurring question around Bob/MATRIX/OpenCore is:

```text
Can capability compound without consuming the routes by which external reality can later correct it?
```

For Bob, the dynamic pipeline remains:

```text
STORE | DETECT | QUERY | RESOLVE | ACT | CONTINUE
```

Lifetime 001 reached QUERY but failed the QUERY->ACT composition. The warrant repair repairs that one transition. Nothing in the repair itself establishes the downstream stages.

## 14. Archive integrity

This handoff branch is an archive/custody object. Its job is to make a new chat able to recover the exact state without relying on conversational memory.

The branch should contain:

- all inherited source/design/plan/result records through the engineering replay;
- this handoff file;
- base64-encoded deterministic gzip of Lifetime-001 raw trajectory;
- base64-encoded deterministic gzip of the 640-step repair replay trajectory;
- a machine-readable handoff manifest containing the same immutable identities.

If any archive copy disagrees with the hashes in the frozen result records, the frozen result-record hashes win and the archive copy is defective.
