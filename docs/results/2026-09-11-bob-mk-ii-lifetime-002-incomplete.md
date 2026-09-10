# Bob Mk II Scientific Lifetime 002 — Incomplete / Invalid Execution Record

Date: 2026-09-11

Standing: `EXECUTION_INCOMPLETE / SCIENTIFIC_INTERPRETATION_PROHIBITED`

This record fossilizes the attempted second authorized Bob Mk II scientific lifetime. It is not a positive result, negative result, or scientific null.

## Provenance

- Design: `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`
- Repaired implementation: `e3be1166ac0a6e7d4abde7856a6d9d53719410fb`
- Authorization: `AUTH-USER-20260911-LIFETIME-002`
- Preflight on exact implementation: `86/86` tests passed; software conformance passed; environment audit remained `ENVIRONMENT_OPPORTUNITY_ONLY` with `bob_executed=false`; minimum partner-prediction gain `0.073279857880927 > 0.02`.

## Attempt 1 — host timeout after scientific start

The authorized lifetime started and emitted `1688` trajectory rows (`step 0..1687`) before the execution host killed the foreground command at its wall-clock timeout.

- Raw prefix bytes: `2,745,751`
- Raw prefix SHA-256: `d8e0f9e760d32911e4ce07a1b22c7f14f68bfa333bc2ff968ac7b1852eb85d18`
- Deterministic gzip bytes: `136,682`
- Deterministic gzip SHA-256: `ccdbda4b0489d50efbefa056f8cd430d866c468106639842e52f9dd89c6dc5de`

The process did not survive the host timeout. No in-memory checkpoint existed, so direct continuation was impossible.

## Recovery rule frozen before restart

A deterministic recovery attempt was allowed only under the same implementation, configuration, and authorization, with this gate:

`first 1688 replay rows must be byte-for-byte identical to Attempt 1; otherwise STOP and invalidate Lifetime 002`.

No implementation or scientific parameter was changed before this check.

## Attempt 2 — recovery gate failed, then stopped

The recovery process was restarted in the background to remove the host foreground timeout. Once it had replayed beyond row 1688, the frozen prefix comparison failed.

- First differing row: `step 639`, the first structural `CREATE` boundary.
- First differing key: `parameter_fingerprint` only.
- Attempt-1 parameter fingerprint at step 639: `08469697ddaf6ff9a5609f7d9e1869e031dbfddb1b4a8aaa5584c032e16107c8`
- Attempt-2 parameter fingerprint at step 639: `8f817e54b953cef9f5537380bcdb72c32227251acbfb17b62e79d8a5617b0b93`
- All compared discrete decision/custody fields through the first 1688 rows were identical, including chosen candidate, eligible set, decision mode/id, gate outcome/reason, proposal/transaction/evidence/warrant identity, active-edge count, route counts, graph metadata, and action exposure/eligibility.
- Discrete-projection SHA-256 for both prefixes: `1f077dfd2109bcb2b03ac37aeeb4204d0a7316f550226bf940be2491eb0e3f9a`.

Per the frozen recovery rule, the process was killed immediately after the mismatch was established. Because process termination is asynchronous, Attempt 2 had emitted `3633` rows by termination.

- Attempt-2 raw bytes: `6,231,555`
- Attempt-2 raw SHA-256: `35e55aea3c6a0f5dd019d2879162fcd14de383eac13be8745def71691809d541`
- Deterministic gzip bytes: `297,731`
- Deterministic gzip SHA-256: `49d355b675947e2b99d400984232b49a0556f6809c694d69cbc430614be70190`

Attempt-2 rows beyond the failed prefix gate are custody data only and are scientifically uninterpretable.

## Reproducibility defect localized

The implementation constructs a newly created structural edge with `LowRankInterface`, whose `u_factors` and `v_factors` use bare process-global `torch.randn`. The backbone initializes its network under `torch.random.fork_rng` with `torch.manual_seed(network_seed)`, which restores the prior process-global RNG state afterward. Consequently the first later structural edge initialization depends on the process's independently initialized global Torch RNG state.

A separate two-process engineering check reproduced the defect: two fresh processes produced different first-edge digests, while explicitly setting the same global Torch seed immediately before edge creation produced identical edge digests in both processes.

This is an implementation reproducibility defect. It does not establish anything about whether the learned manager values future corrective access.

## Claim ceiling

Lifetime 002 has no scientific endpoint. Its only warranted record is:

**The repaired apparatus reached structural creation, but the authorized run was interrupted by the host and could not be recovered bitwise because structural edge initialization was not deterministically seeded.**

A successor scientific lifetime requires a prospectively frozen deterministic edge-initialization repair, a new implementation identity, fresh verification, and fresh explicit execution authorization.