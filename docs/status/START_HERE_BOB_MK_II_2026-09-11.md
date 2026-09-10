# START HERE — Bob Mk II restart pointer

This is the canonical restart pointer for the ChatGPT conversation that reached its length limit on 2026-09-11.

Read these in order before changing or executing anything:

1. `docs/status/BOB_MK_II_CHAT_HANDOFF_2026-09-11.md`
2. `results/archive/BOB_MK_II_HANDOFF_MANIFEST.json`
3. `docs/results/2026-09-10-bob-mk-ii-lifetime-001.md`
4. `docs/results/2026-09-10-bob-mk-ii-probe-warrant-repair-replay.md`
5. `results/archive/BOB_MK_II_CRITICAL_ROWS.json`

Exact plan lineage correction: the full base implementation-plan SHA is `2259c818db9f911e14b7bb3a519fb684cba023b4`; terminal plan SHA is `38cdaebdc9ad6dec3fc1a4d4c9b279a73ad1c0a6`.

Current executable authority for any future Lifetime 002 is the repaired implementation:

`e3be1166ac0a6e7d4abde7856a6d9d53719410fb`

It is CI verified at `86/86` tests plus `SOFTWARE_CONFORMANCE_ONLY steps=8 scientific_result=NONE`.

Current scientific state:

`LIFETIME_001_EXECUTED / UPSTREAM_HANDOFF_FAILURE`

`WARRANT_CONTINUITY_REPAIR_VERIFIED`

`REPAIR_PREFIX_REPLAY_PASS / NOT_FRESH_SCIENCE`

`LIFETIME_002_NOT_AUTHORIZED / NOT_EXECUTED`

The main intended Mk II question — whether delayed realized feedback can train Bob to prefer legal actions partly because of the future corrective options they leave available — remains **UNTESTED**.

Do not silently fix the still-open probe candidate aliasing/generation obstruction before a prospective freeze.

## Raw trajectory custody note

The two raw trajectory identities are frozen in the result records and manifest. The GitHub connector available in the closing chat can write repository text but cannot transfer a local binary file into GitHub. Therefore the raw compressed bytes themselves are **not embedded in this Git branch**. They were separately persisted in ChatGPT Library at:

- `/Research/RCT/Bob-Mk-II/2026-09-11-handoff/lifetime-001-trajectory.jsonl.gz`
- `/Research/RCT/Bob-Mk-II/2026-09-11-handoff/warrant-repair-prefix-replay-trajectory.jsonl.gz`

Their deterministic gzip hashes are respectively:

- `23b201977545e3839bcaddcf58d0f9ba793047348bdcffe78d83876c65460f2e`
- `708b7c99dadfb85d7954ab8770f9a81cefcb556844b3538d1a88f76b6438dc31`

The corresponding raw JSONL SHA-256 values are:

- Lifetime 001: `b955334bc4c1129dca7fbb0ee8f6c3d5246b760f96cad6e2db185bccac95cf7e`
- Repair prefix replay: `b067eea11a257310332eecd13bf59ef18d0bb74d11ea1e76cdc408fd3a38b0b2`

This note supersedes any sentence in the longer handoff that says a base64 copy was successfully embedded in GitHub; that upload path was unavailable. All interpretation, lineage, code, tests, result records, hashes, and the critical trajectory rows needed to recover the research state are on this archive branch.

## Next legal action

A fresh full Lifetime 002 requires a new explicit authorization bound to `e3be1166...`. Do not reuse the Lifetime-001 authorization.

Suggested user phrase: `authorized lifetime 002`.
