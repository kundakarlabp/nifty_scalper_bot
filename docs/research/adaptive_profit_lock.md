# Adaptive profit-lock research candidate (21 October 2026)

## Safety and status

The canonical `BracketManager` owns all stop orders. This change adds an
**opt-in**, after-canonical ratchet inside
`execution/market_aware_profit_extension.py`; the existing
`BoundBracketManager._apply_trailing_math` invokes it. It is **disabled by
default** (`ADAPTIVE_PROFIT_LOCK_ENABLED=false`), and does not weaken or
replace the exchange stop, maximum daily loss, sizing, hard SL/TP,
reconciliation, or entry/readiness gates.

**Research status:** `RESEARCH_CANDIDATE`, engineering validation only.
Do not enable in production until quote-level, fill-aware chronological replay
and independent holdout support a specific policy, followed by a separately
approved gradual controlled rollout. Passing pytest/CI does not demonstrate
improved profit or expectancy.

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
4. Past 1R, select candidate
   `max(minimum floor, entry + locked_fraction * executable_bid_MFE,
        executable_bid_high - ATR_multiplier * option_ATR)`.
   An unavailable/implausible ATR is omitted, not synthesized. Standard
   values are STRONG: 40%/2.0ATR; NORMAL: 50%/1.5ATR;
   WEAK: 65%/1.0ATR. These are **untested hypotheses**, not optimal values.
5. STRONG requires separate premium, underlying, and market-activity
   evidence plus trend regime. WEAK arises from range/low-activity,
   invalidated entry contract or independently conflicting evidence.
   Repeated correlated indicators within one family are not multiple votes.
   A quality-state change requires **two distinct delivered quotes**.
6. Preserve a buffer below live executable bid of at least
   `max(0.05, 1.5*spread, 0.1*ATR)`. Require a meaningful >=0.50 point or
   0.25% stop improvement. Never lower an already ratcheted stop.
7. Refuse speculative stop changes when execution is pending, broker stop
   status is degraded/unverified, or in live mode without an existing
   exchange protective stop. Route any accepted virtual improvement through
   the same `_ratchet_stop` -> broker-stop sync. Confirmation/modify budget
   are still owned by the canonical broker-stop mechanism.

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

## Config values (off unless explicitly enabled)

| Variable | Research default | Max accepted |
|---|---:|---:|
| `ADAPTIVE_PROFIT_LOCK_ENABLED` | false | boolean |
| `ADAPTIVE_PROFIT_MIN_POINTS` | 1.0 | 3.0 |
| `ADAPTIVE_PROFIT_SLIPPAGE_POINTS` | 0.50 | 3.0 |
| `ADAPTIVE_PROFIT_QUOTE_MAX_AGE_SEC` | 3.0 | 5.0 |
| `ADAPTIVE_PROFIT_MAX_SPREAD_PCT` | 1.5 | 2.0 |

No new scheduled research worker, paid market data, broker credentials,
deployment or automated live policy promotion is introduced.
