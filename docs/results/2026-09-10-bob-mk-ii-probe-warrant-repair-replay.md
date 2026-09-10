# Bob Mk II Probe-Warrant Continuity — Engineering Prefix Replay

Date: 2026-09-10

Standing: `ENGINEERING_REPLAY_PASS / NOT_FRESH_SCIENCE / NO_FULL_LIFETIME`

This record validates only the narrow post-Lifetime-001 repair of the `PROBE -> fresh evidence -> CREATE eligibility` handoff. It deliberately reuses the same public world configuration and seeds and stops after the first formerly broken handoff. It is not a second scientific lifetime and does not authorize one.

## Provenance

- Lifetime-001 frozen result parent: `660774a0154dc184d1896ee76dc4a441d18cae23`
- Repaired implementation: `e3be1166ac0a6e7d4abde7856a6d9d53719410fb`
- Repaired implementation lineage: exactly one commit ahead of the frozen Lifetime-001 result record
- Frozen original executed implementation: `095505276faf0faaad02d71241f1d881999396f8`
- Design remains: `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`
- GitHub Actions for repaired implementation: `86 passed`; `python -m rct.bob.cli validate` returned `SOFTWARE_CONFORMANCE_ONLY ... scientific_result=NONE`

The repair changes only the probe-acquisition-warrant handoff and its focused tests. It does not change the world, 38-feature schema, valuation learner, candidate generator, costs, exploration schedule, action vocabulary, or one-action-per-boundary sequencing.

## Repair contract

The repaired path is:

`qualified PROBE -> typed ProbeAcquisitionWarrant -> ProbeEvidence(warrant_ref) -> CREATE while evidence+warrant are fresh`.

`ProbeEvidence` does not itself become warrant or authority. The authoritative structural gate issues the warrant only after the existing PROBE qualification passes. CREATE requires matching fresh unconsumed evidence and warrant. Successful CREATE consumes both.

The repair intentionally does not address the separate probe-candidate aliasing/generation issue observed in Lifetime 001.

## Replay custody

The engineering replay ran ordinary steps `0..639` only.

- Same private schedule reproduced: `W1=2476`, `W2=2478`, `W3=1632`
- Replay trajectory rows: `640`
- Raw replay trajectory SHA-256: `b067eea11a257310332eecd13bf59ef18d0bb74d11ea1e76cdc408fd3a38b0b2`
- Raw replay trajectory bytes: `903,530`
- Deterministic gzip SHA-256 (`gzip -n -9`): `708b7c99dadfb85d7954ab8770f9a81cefcb556844b3538d1a88f76b6438dc31`
- Deterministic gzip bytes: `50,330`

## Formerly broken seam

At step `575`, the replay reproduced the same candidate set as Lifetime 001:

- `PROBE:C->D`
- `PROBE:A->D`
- `ABSTAIN`

The manager again selected `PROBE:A->D`. That action produced:

- evidence: `E000001`
- acquisition warrant: `W000001`
- warrant endpoints: `A -> D`
- warrant/evidence validity: steps `575..639`, inclusive

At step `639`, the eligible set was:

- `CREATE:A->D:E000001`
- `ABSTAIN`

The manager selected `CREATE:A->D:E000001`. The gate returned:

- `allowed = True`
- `reason = FORMABLE`
- transaction: `T000001`

After the transaction:

- active-edge count = `1`
- `E000001` consumed = `True`
- `W000001` consumed = `True`

Thus the narrow engineering criterion passed:

`same step-575 PROBE -> CREATE is formable at the next boundary despite loss of the original three-window trigger`.

The fact that CREATE was also selected is downstream behavior observed in this same-seed replay; it is not an independently randomized scientific endpoint.

## Claim ceiling

This replay supports only:

**The typed acquisition-warrant repair removes the specific temporal handoff obstruction observed in Lifetime 001 under the same-seed prefix replay.**

It does not establish:

- that the repaired agent improves task performance;
- that the created `A->D` edge is useful;
- that Bob will adapt appropriately in W2 or W3;
- that future-access features affect action selection;
- that learned valuation preserves corrective access;
- safe forgetting, corrigibility, or general self-revision.

A full successor W1->W2->W3 lifetime requires a fresh explicit execution authorization bound to repaired implementation `e3be1166ac0a6e7d4abde7856a6d9d53719410fb`.
