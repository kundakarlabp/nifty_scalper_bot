# Status log hygiene

## Live execution review — 10 September 2026, Asia/Kolkata

Source register: production snapshot at 14:07 IST on `c5de7bd8`, and
14:05–14:09 order logs retrieved through the existing Supabase/AppDeploy relay.
Repeated prebroker rejections reported net R:R 0.96–1.07 against 1.50. A selected
contract changed during one attempt; transient overload rejected another.
Follow-up health after market close reports deployed `1987552c`, authenticated
broker, completed reconciliation, no unprotected positions, and `market_closed`.
Some background logs also report risk halt/reconciliation age; next-session
readiness must be checked before declaring live execution verified.

Decision: retain stop risk, quantity sizing, market hours and final net-R:R gates.
The existing cost-repair helper was given strategy lots (1) while its caller
already resolved broker units (65). Use units only on the calculation copy;
return the original strategy lot count. No threshold or target-cap changes.
The fix is based on authoritative `1987552c5e3470f4523f3deea75f13a2770b91d6`.

Validation: one- and two-lot regressions fail before the correction and pass
after it. The deliberately narrow setup remains rejected under a +0.35R cap.
44 adjacent tests and two subtests pass; production compilation passes.
New-test Ruff/Black and changed-line Black pass. Existing production-file
formatting/typecheck debt is outside this quantity correction.

The full-suite attempt hit the unrelated timing/log-throttle assertion in
`test_full_one_tick_timing_covers_early_normalization_return`; it passes alone.
The remaining suite is being rerun and required CI must pass before merge.

Owner: repository maintainer/assistant. Status: locally corrected; PR gates
must pass before merge. Next review: deployed SHA and first
market-open execution result. No promised order or profit. Separate remaining
defect: generic `risk_manager_blocked` hides the detailed risk reason from the
runner's existing deterministic rejection cooldown. No NIMS-Chrome interface
or other cross-repository behavior is affected.

Non-gating context-symbol hydration misses should not be counted as operator errors. Reconciliation display should reflect a completed reconcile marker when available in bounded runtime logs.
