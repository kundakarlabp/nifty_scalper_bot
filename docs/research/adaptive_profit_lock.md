# Adaptive profit-lock research candidate (21 October 2026)

## Safety and status

The canonical `BracketManager` owns all stop orders. This change adds an
**broker-live-default**, after-canonical ratchet inside
`execution/market_aware_profit_extension.py`; the existing
`BoundBracketManager._apply_trailing_math` invokes it. For a **verified
real-broker LIVE** manager it is **enabled by default**, while paper,
shadow and historical replay remain disabled unless explicitly opted in.
To disable adaptive trailing immediately on subsequent evaluations set
`ADAPTIVE_PROFIT_LOCK_ENABLED=false` in the runtime environment and
restart/reload through the validated release process. To force it on in
replay, set `ADAPTIVE_PROFIT_LOCK_ENABLED=true`. It does not weaken or
replace the exchange stop, maximum daily loss, sizing, hard SL/TP,
reconciliation, or entry/readiness gates.

**Evidence status:** `LIVE_DEFAULT_UNVALIDATED_EXIT_POLICY`.
Live default activation was explicitly requested without a completed
fill-aware historical holdout. It is an **experimental live financial risk**
and is not evidence of improved profit or expectancy. An existing exchange
protective stop is still mandatory for live adaptive ratchets. The
`ADAPTIVE_PROFIT_LOCK_ENABLED=false` operator override allows reverting
to the previous market-aware trailing policy without removing canonical
protection. The broker-stop owner still determines whether a submitted
modification is confirmed; a virtual lock is not a guaranteed fill.

## Design

1. Use only a recent (<3-second default) option top-level **bid with visible
   quantity >= remaining position quantity**. Require a valid ask and spread
   <=1.5% (tunable within a bounded range). The last traded price and
   incomplete/stale quotes cannot establish executable peak profit.
2. Record a monotonic water mark of qualifying executable bids per trade
   (`_adaptive_profit_executable_high` in canonical bracket trailing config).
   Start cost-adjusted profit locking after 0.6 initial-risk units (0.6R).
3. Minimum proposed long-option stop is
   `entry + estimated roundtrip cost/unit + stressed exit slippage/unit
    + max(fixed net profit points, 0.10R)`.
   Initial research values: fixed profit 1.0 premium point/unit, slippage
   0.50 premium point/unit. Both are bounded, do not imply guaranteed fills.
4. Past 1R, determine a **single giveback allowance**, not the tightest
   of independent percentage and ATR stop candidates:
   `allowance = max((1-lock_fraction)*MFE, regime_ATR*ATR,
   3*median_abs_recent_bid_change)`; then
   `stop = max(cost_adjusted_floor, executable_bid_peak - allowance)`.
   The allowance is widened by volatility, never inadvertently tightened
   by a competing ATR expression. Unavailable or unproven-fresh ATR is omitted,
   not synthesized. STRONG uses 40%/2ATR; NORMAL 50%/1.5ATR;
   WEAK 65%/1ATR, **research hypotheses only**.
5. STRONG requires separate premium, underlying, and market-activity
   evidence plus trend regime. WEAK arises from range/low-activity,
   invalidated entry contract or independently conflicting evidence.
   Correlated indicators within one family are not multiple votes.
   A quality-state change requires **two distinct market assessments**,
   not multiple ticks using the same cached assessment.
6. Track a bounded rolling history of full-size executable bids; use
   median absolute successive bid change over a bounded eight-second
   window as a short-horizon noise estimate.
   With peak age >=8 seconds, WEAK quality and independent premium plus
   underlying/activity deterioration, increase retention by up to 10%
   (maximum 75%). Preserve a live-bid buffer of at least
   `max(0.05, 1.5*spread, 1.5*short_bid_noise)`. Require >=0.50
   point or 0.25% stop improvement. Never lower a ratcheted stop.
7. Refuse speculative stop changes when costs are not positively verified,
   execution is pending, broker stop status is degraded/unverified, or
   live mode lacks an exchange protective stop. Route any accepted virtual
   improvement through the same `_ratchet_stop` -> broker-stop sync.
   Canonical fallback, market-aware and adaptive proposals now share a
   **single broker sync attempt per position evaluation**. When the
   research policy is enabled, it replaces the legacy standalone market-aware
   tightening proposal; the canonical baseline and target-extension safety
   checks remain. Other threads,
   other positions and protection placement are not deferred. Confirmations
   and modification budgets remain with the broker-stop owner.

## Replays before activation

Compare *unchanged* canonical exits against the adaptive candidate using
recorded production-received `ReplayArchive` ticks with verified contract,
quote timestamps, depth, and order quantity, and the same entries, costs and
slippage assumptions. Use the existing
`backtesting/exit_policy_research.py` for preliminary quote-level overlays,
and the full production-composition runtime replay for final engineering
parity. Missing executable bid, no complete interval, broker SL-limit gap,
uncertain fill or surviving counterfactual is **UNRESOLVED**, never a
backfilled success.

Evaluate separate TRAIN/VALIDATION/UNTOUCHED TEST dates and compare net
expectancy, profit factor, drawdown, tail losses, MFE capture, early exits,
missed continuation, stop modification budget, pending-state exposure,
liquidity coverage, and realized slippage. Stress spreads and worsening
fills by >=0.5 and >=1.0 premium points, stop-limit gaps and reconnects;
segment by trend/range, strategy, time of day and risk per trade. Avoid
selecting the best configuration from one day or the 12 complete VWAPPro
trades available in October 2026.

## Config values (enabled by default only in verified broker LIVE)

| Variable | Research default | Max accepted |
|---|---:|---:|
| `ADAPTIVE_PROFIT_LOCK_ENABLED` | real broker LIVE: true; other modes: false | operator true/false |
| `ADAPTIVE_PROFIT_MIN_POINTS` | 1.0 | 3.0 |
| `ADAPTIVE_PROFIT_SLIPPAGE_POINTS` | 0.50 | 3.0 |
| `ADAPTIVE_PROFIT_QUOTE_MAX_AGE_SEC` | 3.0 | 5.0 |
| `ADAPTIVE_PROFIT_MAX_SPREAD_PCT` | 1.5 | 2.0 |
| `ADAPTIVE_PROFIT_STALL_SECONDS` | 8.0 | 30.0 |

No new scheduled research worker, paid market data, broker credentials,
deployment or automated live policy promotion is introduced.
