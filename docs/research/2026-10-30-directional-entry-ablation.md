# 30 October 2026 — staged directional entry research (no LIVE promotion)

## Scope and base

Repository: `kundakarlabp/nifty_scalper_bot`. This change belongs to a
separate research branch; the existing strategy manager, candidate selection,
risk checks, single order path, stops, exits, adaptive profit lock and
broker reconciliation remain unchanged.

**Priority 1 — trend pullback/continuation.** The existing VWAPPro strict
continuation rule already requires a recent thesis, completed option breakout,
current underlying TREND/direction and confirmed premium activity. It labels
its successful proof `vwap_continuation_proof` as either
`retest_rebreak` or `momentum_breakout`. These are not independent signals.

The one-variable ablation is `VWAP_RESEARCH_RETEST_REBREAK_ONLY=true` with
`EXECUTION_MODE=SHADOW` and
`VWAP_STRICT_CONTINUATION_ENABLED=true`.
Only `vwap_continuation` is affected, and only when it lacks an already
validated `retest_rebreak` proof. The feature is disabled by default and
ignored in LIVE/PAPER even when accidentally configured. It does not modify
`vwap_reclaim`, `vwap_reclaim_momentum`, SMC, ORB, broker or exit logic.
The existing `vwap_research_retest_rebreak_required` rejection identity
separates this experiment from invalid market data and non-admission.

This is an **eligibility ablation**, not proof of improved profitability. A
trade suppressed by the retest-only arm must remain in the opportunity ledger,
with time-stamped original option/futures context and reason.

To run the archived paired study, set
`VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY=true` in a **SHADOW**
research process and invoke the existing `run_vwap_comparison` job.
It adds the two arms `strict_continuation_either_proof` and
`strict_continuation_retest_rebreak_only` to the nine baseline candidates.
Both explicitly enable the existing strict continuation proof; the retest-only
toggle is their sole policy difference. Compare at 10/25/50 bps per-side
slippage, with the existing research simulator's net RR and eligibility rules.
The research runner clears inherited experimental values before each arm and
restores the original environment afterward. It does not certify execution
parity; the existing report's `promotion_eligible=false` is retained.

**Priority 2 — selective ORB.** Do not create another ORB implementation.
The repository already exposes separate momentum/retest outcome metadata
(`entry_branch`, `retest_confirmed`), opening range width / ATR,
underlying penetration / ATR, volume ratio, breakout timestamps and event
identity. Existing `scripts/research_orb_iterative.py`,
`scripts/research_orb_multiyear.py` and
`src/nifty_scalper_bot/backtesting/strategy_research.py` already compare
`ORB_MOMENTUM_BRANCH_ENABLED=false` with the baseline. Reuse those
research profiles, rather than enabling new LIVE rules or duplicating them.

## Evidence required before approval

1. Verify exact completed candle provenance, same-session original VWAP
   thesis and point-in-time eligible weekly CE/PE contract selection.
2. Use identical session/contract opportunity universes for the baseline
   and candidate; record suppressed opportunities and no-fills.
3. Evaluate the *option* after brokerage, current taxes, spread, partial
   fills, latency, adverse selection and slippage; report trade count,
   rejection counts, expectancy, drawdown, expected shortfall, capture
   rate, entry delay and opportunity cost.
4. Separate momentum-only, retest-only, and any both-proof setups. Keep
   fixed sizing, exit policy, risk controls and capital across arms.
5. Use chronological out-of-sample periods, session-blocked uncertainty
   and an untouched holdout; log every attempted configuration.
6. Backtest reports with missing bid/ask, historical option identity or
   incomplete futures opening range are *unresolved*, not winning evidence.
7. Pass focused tests and all required CI, then prospective shadow
   comparison and broker-flat release checks before considering any LIVE
   change. No strategy may be promoted merely because a flag exists.

## Rollback

Leave `VWAP_RESEARCH_RETEST_REBREAK_ONLY=false` or unset. There is no
production change to roll back. Keep the existing
`VWAP_STRICT_CONTINUATION_ENABLED=true` default untouched. Existing ORB
baseline settings remain unchanged.

## Release status

Engineering research branch only; no new backtest results, profitability
verification, deployment, or change to the production engine claimed.
