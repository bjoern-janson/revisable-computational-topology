# Bob Mk II implementation-plan self-review amendment

**Repository:** `bjoern-janson/revisable-computational-topology`

**Base implementation plan commit:** `2259c818db9f911e14b7bb3a519fb684cba023b4`

**Approved design:** `716118147c5ec8925eaa1c3ddc3b5f500eb51c29`

**Standing:** `NORMATIVE_PLAN_AMENDMENT / IMPLEMENTATION_NOT_STARTED / SCIENTIFIC_EXECUTION_UNAUTHORIZED`

This amendment fixes ambiguities found during the required plan self-review. It changes no approved design claim. Where it conflicts with `docs/superpowers/plans/2026-09-10-bob-mk-ii-inquiry-valuation.md`, this amendment controls.

## 1. Exact manager pressure, qualification, and cooldown

Freeze manager summary updates as follows.

Per-step module and existing-edge EMAs use:

```python
EMA_BETA = 0.95
ema_next = EMA_BETA * ema_prev + (1.0 - EMA_BETA) * observed
```

At every 64-step structural boundary, compute per-target means over the just-completed 64 ordinary steps. For target `j`:

```python
local_loss_slope_j = current_window_local_mean_j - previous_window_local_mean_j

discovery_pressure_j = (
    local_loss_ema_j
    if t >= manager_warmup_steps and abs(local_loss_slope_j) < 0.002
    else 0.0
)

revision_pressure_j = (
    max(0.0, full_loss_ema_j - local_loss_ema_j)
    + max(0.0, -sum(edge.contribution_ema for edge in incoming_active_edges(j)))
)

inquiry_pressure_j = max(discovery_pressure_j, revision_pressure_j)
```

Maintain two target-specific qualification counters. A counter increments when its pressure is strictly greater than `0.05`; otherwise it resets to zero:

```python
inquiry_qualifying_windows_j = consecutive_windows(inquiry_pressure_j > 0.05)
revision_qualifying_windows_j = consecutive_windows(revision_pressure_j > 0.05)
```

An action's pressure prerequisite is:

```text
PROBE / CREATE       -> target inquiry_qualifying_windows >= 3
MODIFY / DORMANT     -> target revision_qualifying_windows >= 3
RETIRE               -> target revision_qualifying_windows >= 3 + retirement support gate PASS
REOPEN               -> target revision_qualifying_windows >= 3
ABSTAIN              -> none
```

`ExistingEdgeEvidence` is emitted at every structural boundary after warmup for every currently active edge and records exact generation, target revision pressure, current contribution EMA, usage EMA, active rank, created step, and expiry step. Emission is observation, not permission.

`ReopenEvidence` is emitted at every structural boundary after warmup for every currently VALID handle whose `historical_support > 0`; it records current target revision pressure, exact generation+dormancy version, historical support, liability, created step, and expiry step. Emission is observation, not permission.

Both evidence types expire at `created_step + 64`. Gate qualification still controls whether they can support a structural action.

There is exactly one structural-edit cooldown window after any accepted CREATE, MODIFY, DORMANT, RETIRE, or REOPEN:

```python
STRUCTURAL_COOLDOWN_WINDOWS = 1
```

During that next structural boundary, structural mutations are not formable. PROBE and ABSTAIN remain formable if their own prerequisites hold. A PROBE or ABSTAIN does not start/reset structural cooldown.

## 2. Exact candidate-generation scope

At a structural boundary, construct the legal absent-pair pool only from pairs whose target satisfies `inquiry_qualifying_windows >= 3`. The pool contains no pair-specific evidence or score. Sample at most two pairs uniformly without replacement using the independent manager RNG stream seeded `31`.

If the qualified pool is empty, generate no PROBE relation candidate. Do not substitute unqualified pairs merely to fill two slots.

This implementation therefore establishes only bounded system-generated candidate presentation plus learned action valuation. It does not establish learned candidate generation.

## 3. Exact 38-coordinate feature formulas

All action feature tensors are CPU `torch.float64`. Any field that is inapplicable to an action is exactly `0.0` unless specified below. No missing-value learned embedding exists.

Let:

```python
EPS = 1e-6
CLIP = 2.0
B = edit_budget_per_window  # 0.50
```

and:

```python
def signed_ratio(x, scale):
    return max(-CLIP, min(CLIP, x / max(abs(scale), EPS)))
```

Coordinates are frozen as follows.

### `phi_now`

```text
00 bias = 1.0
01 pressure_ratio = signed_ratio(action-relevant target pressure, target full_loss_ema)
02 contribution_ratio = signed_ratio(edge contribution_ema, target full_loss_ema); 0 if no existing edge
03 usage_ema = edge usage_ema clipped to [0,1]; 0 if no existing edge
04 probe_gain_ratio = signed_ratio(probe_evidence.gain, probe_evidence.baseline_mse); 0 without ProbeEvidence
05 evidence_freshness = clip((expires_step-current_step)/64, 0, 1); 0 with no evidence prerequisite
06 rank_fraction = active_rank/8; 0 if no edge/handle rank applies
07 probe_budget_fraction = current remaining probes / 2
08 edit_budget_fraction = current remaining structural edit budget / 0.50
```

For coordinate 01, action-relevant target pressure means inquiry pressure for PROBE/CREATE, revision pressure for MODIFY/DORMANT/RETIRE/REOPEN, and `0` for ABSTAIN.

### `phi_cost`

Define deterministic candidate charges from authoritative config/current state. Then:

```text
09 probe_charge_fraction = probe_charge / B
10 edit_charge_fraction = structural edit charge / B
11 maintenance_delta_fraction = clip((post_action_maintenance - pre_action_maintenance)/B, -2, 2)
12 reopen_liability_fraction = reopen liability / B
13 total_immediate_charge_fraction = (probe_charge + edit_charge + reopen_liability) / B
```

Maintenance is deliberately not included again in coordinate 13 because coordinate 11 separately represents its change and realized horizon maintenance enters `Y_t` directly.

### `phi_future`

The route cohort is the current persistent historical route cohort before the candidate. Dry runs do not admit new consequential commitments; they only update deterministic retention/reachability metadata for already-admitted routes.

```text
14 route_profile_defined = 1 if current route_count > 0 else 0
15 delta_retained = C_retained_after - C_retained_before; 0 if route_count == 0
16 delta_reachable = C_reachable_after - C_reachable_before; 0 if route_count == 0
17 post_probe_headroom = post_action_remaining_probes / 2
18 post_formable_nonretire_class_fraction = number of distinct formable classes among
   {PROBE, CREATE, MODIFY_UP, MODIFY_DOWN, DORMANT, REOPEN, ABSTAIN} after dry run / 7
19 delta_valid_handle_fraction = clip((valid_handles_after-valid_handles_before)/12, -1, 1)
20 reachable_route_concentration_after = max_c reachable_routes_for_commitment(c) / total_reachable_routes_after;
   0 if total_reachable_routes_after == 0
21 produces_evidence_flag = 1 for PROBE, else 0
```

For coordinate 18, the dry run may apply deterministic bookkeeping only. It may not assume the content/sign of a future PROBE result, neural learning after CREATE, or future world observations. A PROBE can therefore reduce budget/headroom and set coordinate 21, but cannot make CREATE formable in the same dry run from an answer that does not yet exist.

### `phi_lineage`

```text
22 edge_age_fraction = min(edge_age/2048, 1); 0 if no edge/handle
23 generation_fraction = min(generation/8, 1); 0 if none
24 dormancy_version_fraction = min(dormancy_version/8, 1); 0 if none
25 historical_support_ratio = clip(historical_support/max(target full_loss_ema,EPS), 0, 2); 0 if NONE
26 time_since_related_action_fraction = min(steps_since_last_action_on_same ordered endpoint pair/256, 1);
   1 if the pair has no prior action, 0 for ABSTAIN
27 accepted_related_fraction = min(accepted actions on same ordered endpoint pair/8, 1); 0 for ABSTAIN
28 rejected_related_fraction = min(rejected proposals on same ordered endpoint pair/8, 1); 0 for ABSTAIN
29 prior_correction_exercised = 1 iff a CorrectionOutcome exists for a route tied to this exact edge generation/handle family; 0 otherwise
```

### `phi_action`

Coordinates `30..37` are one-hot in this exact order:

```text
30 PROBE
31 CREATE
32 MODIFY_UP
33 MODIFY_DOWN
34 DORMANT
35 RETIRE
36 REOPEN
37 ABSTAIN
```

Exactly one is `1.0`; the other seven are `0.0`. No feature interactions are appended.

## 4. Exact cost timing in the delayed return

A structural decision is timestamped at boundary step `t` after the ordinary prediction/update work for that step. Any probe/edit/reopen charge caused by that decision is assigned to return time `t`. Maintenance cost for already-active/post-action structure is charged on each ordinary step for the structure actually used during that step.

For a decision at `t`, the ledger accumulates exactly timestamps:

```text
t, t+1, ..., t+255
```

and becomes eligible to mature when the ordinary clock reaches `t+256`. Processing timestamp `t+256` first matures the prior row and does **not** include timestamp `t+256` in that row.

This removes the off-by-one ambiguity while preserving the approved `H=256`, `gamma=1` design.

## 5. Self-review closure

The self-review checked:

```text
spec coverage       PASS after this amendment
placeholder scan    PASS: no TBD/TODO/implement-later instructions
type/interface pass PASS at plan level
feature edge cases  CLOSED by Section 3 above
pressure/cooldown   CLOSED by Section 1 above
return off-by-one   CLOSED by Section 4 above
```

The plan remains non-executing. No implementation, tests, audit, conformance run, W1→W2→W3 lifetime, or scientific result has been produced by this amendment.