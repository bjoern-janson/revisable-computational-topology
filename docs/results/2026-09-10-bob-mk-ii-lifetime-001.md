# Bob Mk II Scientific Lifetime 001 — Frozen Result Record

Date: 2026-09-10

Standing: `EXECUTED / INTERPRETED / UPSTREAM_HANDOFF_FAILURE`

This record fossilizes the first authorized Bob Mk II scientific lifetime. It does not modify the frozen implementation and does not authorize a rerun.

## Provenance

- Design: `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`
- Terminal implementation executed: `095505276faf0faaad02d71241f1d881999396f8`
- Authorization: `AUTH-USER-20260910-FIRST-LIFETIME`
- World schedule disclosed after completion: `W1=2476`, `W2=2478`, `W3=1632`
- Total ordinary steps / trajectory rows: `6586`
- Process exit: `0`
- Raw trajectory SHA-256: `b955334bc4c1129dca7fbb0ee8f6c3d5246b760f96cad6e2db185bccac95cf7e`
- Raw trajectory bytes: `10,396,380`
- Deterministic gzip SHA-256 (`gzip -n -9`): `23b201977545e3839bcaddcf58d0f9ba793047348bdcffe78d83876c65460f2e`
- Deterministic gzip bytes: `526,423`

The raw trajectory is an append-only JSONL custody object. Its hash, not this interpretation, is the identity of the observed run.

## Observed terminal facts

- Manager decisions: `102`
- Chosen actions: `100 ABSTAIN`, `2 PROBE`
- Structural transactions: `0`
- Maximum active edges: `0`
- Maximum correction-route count: `0`
- Matured valuation decisions at terminal step: `98`
- Pending valuation decisions at terminal step: `4`
- Structural fingerprint: constant for the entire lifetime

The two probes occurred at ordinary steps `575` and `4863`, each costing `0.02`.

At step `575`, eligible candidates were `PROBE:C->D`, `PROBE:A->D`, and `ABSTAIN`. The two probe candidates had byte-for-byte equal 38-coordinate feature vectors and equal predicted values (`-146.92291422887894`). The deterministic tie break selected `PROBE:A->D`. Audit-private post-run knowledge identifies `C-D` as the relevant W1 pair.

At step `4863`, eligible candidates were `PROBE:A->C`, `PROBE:B->C`, and `ABSTAIN`. Again the two probe feature vectors and scores were identical (`-188.85963957269473`). The selected action was `PROBE:B->C`. Audit-private post-run knowledge identifies `C-D` as the relevant W2 pair, so the useful partner was not in the generated two-candidate set.

At the immediately following structural boundaries (`639` and `4927`), `ABSTAIN` was the only eligible candidate. The acquired probe evidence was still within its declared one-window freshness interval, no structural edit had occurred, and no active endpoint conflict existed. Under the frozen gate, `CREATE` separately re-requires three currently qualifying inquiry-pressure windows. Thus the run localizes a temporal warrant-composition failure: evidence can be legally acquired at one boundary yet become unusable for `CREATE` at the next because the acquisition trigger is re-tested rather than carried by the fresh evidence.

## Loss telemetry

Mean full task loss:

- W1: `0.7789260341224647`
- W2: `0.7287719342331428`
- W3: `0.7206823990917673`

For the regime transition diagnostic only:

- trailing 256-step W2 mean: `0.7042340680491179`
- first 256-step W3 mean: `0.7168430557940155`

These are descriptive task-loss observations, not a causal estimate of structural adaptation. Ordinary neural parameter learning continued throughout the lifetime.

## Delayed-return observations for the two probes

- Decision `D000008` (step 575): decision-time prediction `-146.92291422887894`; matured 256-step realized return `-199.8969029676914`; residual `-52.973988738812466`.
- Decision `D000075` (step 4863): decision-time prediction `-188.85963957269473`; matured 256-step realized return `-178.2621405315399`; residual `10.597499041154833`.

These are realized associations under the continuing policy. They are not causal effects of probing; no matched counterfactual exists and the delayed-return windows overlap other continuing dynamics.

## Scientific interpretation

The strongest warranted interpretation is:

`DETECT/GENERATE -> QUERY occasionally succeeds -> QUERY !-> structural ACT`.

Lifetime 001 is therefore an **upstream generation/detection/warrant-composition failure**, not evidence against learned future-corrective-access valuation. Bob never acquired a structural commitment whose retained or reachable correction routes could later matter. The route-dependent future-access feature block was consequently never substantively exercised.

This run does **not** establish or refute:

- that feedback can teach Bob to prefer structural actions that preserve future corrective access;
- safe forgetting;
- general corrigibility;
- query invention;
- representation invention;
- a MATRIX safe-quotient frontier;
- general self-revision or recursive self-improvement.

## Next legal scientific move

Preserve this run unchanged. A successor experiment may narrowly repair the `PROBE -> fresh evidence -> CREATE eligibility` handoff while holding the world, learned valuation, feature schema, costs, action vocabulary, exploration schedule, and one-action-per-boundary sequencing fixed. Any such repair requires a separate prospective freeze and a new implementation identity before execution.
