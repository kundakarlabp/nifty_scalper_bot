# Regime Gate Runbook

## Quick Commands
- **Snapshot:** `/regime` – latest regime, adjustments, and filter stats.
- **Diagnostics:** `/regimediag` – JSON payload covering history and decisions.
- **Bypass:** `/regime_bypass status|on|off|toggle` (alias `/regimebypass`).
- **Dry-run plan:** `/plan SYMBOL SIDE QTY` – includes regime gate outcome.

## Common Block Reasons
- `regime_unavailable` – no snapshots yet; confirm the detector tick feed is active.
- `regime_stale` – latest snapshot is older than `stale_after_seconds`.
- `confidence_below_floor` – detector confidence under `min_confidence`.
- `regime_block_event` / `regime_block_volatile` – confidence above block thresholds.

## Operator Knobs
- **Environment toggles** (set before boot):
  - `REGIME_MIN_CONFIDENCE` (default `0.40`)
  - `REGIME_STALE_AFTER_SEC` (default `300`)
  - `REGIME_BLOCK_EVENT` (default `0.80`)
  - `REGIME_BLOCK_VOLATILE` (default `0.95`)
  - `REGIME_FAIL_CLOSED` (`0` or `1`)
  - `STRATEGY_ENFORCE_BLOCKLIST` (`0` or `1`)
- **Runtime API** (via `ctx.market_regime_manager`):
  - `min_confidence`
  - `stale_after_seconds`
  - `block_thresholds['event'|'volatile']`
  - `set_regime_filter_bypass(True|False)`
  - `toggle_regime_filter_bypass()`

## Forced Snapshot Injection
```python
import time
from nifty_scalper_bot.core.market_regime import RegimeSnapshot
snap = RegimeSnapshot("NIFTY", "event", 0.92, "forced-test", time.time(), {})
ctx.market_regime_manager.ingest_snapshot(snap)
```

## Runbook – Bot Not Trading
1. `/regime` – verify a fresh snapshot and confirm the regime is benign.
2. `/regimediag` – inspect decision history and confirm `last_decision=allow`.
3. `/regime_bypass off` – ensure the central gate is active (no forced bypass).
4. `/plan NIFTY BUY 1` – expect `risk_gate=True`, `regime_gate=True`.
5. Re-enable strategies if needed: `/strategy_enable <name>`.
6. Check `/tail 200` – confirm no repeated `regime_blocklist_veto` or errors.
7. If still blocked, adjust strategy thresholds before loosening regime guards.


## Stable regime state

The detector publishes raw observations; `MarketRegimeManager` owns the accepted
stable regime used by strategy scoring and gating. Ordinary regime changes require
consecutive confirmations (default `REGIME_TRANSITION_CONFIRMATIONS=2`).
High-confidence regimes that already meet a configured safety block threshold are
accepted immediately so hysteresis can never delay a protective block.

Freshness, confidence-floor, and unknown-regime safety checks use the freshest raw
observation. Diagnostics expose both `current` (stable) and `raw` state plus any
pending transition.

`StrategyRunner` consumes this accepted stable state for its existing strategy-regime
admission rules; it no longer recomputes a second ADX/ATR/VWAP-slope regime. The
runner's coarse ATR low-volatility check remains a separate execution-safety guard,
not a regime classifier or direction authority.


## Completed-bar confirmation and local structure research

Indicator-driven TREND/RANGE transitions require distinct adjacent completed
one-minute evidence. Repeated refreshes of the same candle cannot confirm a
new directional regime. Missing or gapped history cannot manufacture a
confirmation. Raw safety observations retain their immediate veto behavior.
Explicit injected observations retain their existing external snapshot contract.

RangeVWAPReclaim observes local range structure before session-VWAP setup
filters: 25-bar efficiency, ATR-normalised 30-bar width, rolling-center slope,
adjacent-window persistence and two-sided rejection. A continued completed
breakout invalidates the local range immediately. The 60-bar directional
context is absent until sufficient completed history exists. The descriptive
evidence fraction is not a calibrated probability.

The existing journal and post-market report record distinct completed futures
minutes, first-seen session excursions, and observed frozen local/session VWAP
futures target touches. Prior dated reports supply a clock-matched ATR-bps
median only when at least five prior sessions have that exact IST minute.
This is research evidence, not a live override or an option fill simulation.
The existing canonical regime eligibility, risk ceiling, protective
stops and broker reconciliation are unchanged.

The live RANGE route consumes the verified local-breakout veto and, when a
60-bar opposing directional context is proven (efficiency above the existing
range ceiling and displacement at least 1.5 current ATR), requires confirmed
same-side local reclaim and the existing conditional-flow and net-payoff
buffers. Missing 60-bar evidence is not relabeled as RANGE. This conservative
engineering guard is uncalibrated; it does not admit new trades during a
canonical TREND or weaken any cash-risk or broker gate. Clock-matched ATR
and alternative target outcomes remain research-only pending evidence.

## Live evidence integrity and nested-range pilot

Production uses the canonical NIFTY spot instrument and a single completed-bar
indicator snapshot, with finite OHLC, contiguous current-session minutes and an
actual completed-bar timestamp. Active futures volume corroborates only the
same minute and unchanged selected contract; absent index or futures volume
remains unavailable. A missing, stale, conflicting or gapped observation
cannot establish or refresh permission. Startup and gaps require the configured
number of distinct adjacent observations, while raw safety vetoes remain
binding. The live path does not accept quote-driven competing directional
snapshots or permit diagnostic bypasses to waive required evidence.

The runner reuses the current canonical gate immediately before entry submission
and rejects a changed or unavailable decision regime without calling the broker.
Protective exits remain independent of entry regime gating.

ADX is explicitly on its 0–100 scale; legacy trend_score ratios remain a separate
compatibility input. Nonfinite values and boolean numeric impostors do not prove
regimes. Volume corroborates directional evidence, never establishes TREND alone.
Completed ATR in basis points and 60-bar spot displacement/efficiency are carried
through the existing regime context alongside the futures-local assessment.

The additional `ENABLE_RANGE_VWAP_LOCAL_TREND_LIVE` pilot defaults to false. It
also requires the existing live RangeVWAP opt-in, raw/stable TREND agreement,
canonical gate permission, two confirmed local observations, aligned spot and
futures broad context, same-side local reclaim and all existing session-VWAP,
quote, flow, affordability and broker gates. The minimum 1.75 net R:R (or the
existing stronger configured threshold) survives final order repricing; the
773-rupee ceiling is unchanged. Do not arm this new route without cost-aware
chronological evidence. Existing clock-matched prior-session ATR research is
not a calibrated live threshold; hard volatility thresholds remain binding.

Independent volatility-hazard confidence is gated even when the directional
score wins classification. The ATR point thresholds are additionally scaled
against an explicit 25,000-index reference (equivalent to 4.8/8.8/14 bps),
without reducing the original point-based protection. This is a conservative
normalisation convention, not empirically optimised time-of-day thresholds.

## Intermediate ADX, denied observations and intrabar safety

The selected-contract observation hook records completed futures evidence even
when the canonical gate denies entry, with an explicit decision timestamp and
no synthetic execution quote. One active-basket snapshot binds the selected
option and future; broker-owned lot metadata permits later full-lot tape labels
without inventing model risk. Missing completed history is retried rather than
consuming the current minute's research slot. The live gate continues to deny
UNKNOWN. In the ADX interval between the existing chop and trend thresholds,
30 completed spot bars provide descriptive efficiency, ATR-normalised move
and center crossings. Strong two-sided or directional evidence is recorded as
an experimental RANGE/TREND candidate, not as a calibrated permission.

The canonical gate also checks the cached, timestamp-verified NIFTY spot quote
against the completed reference price using the existing event-move threshold
(normalized without weakening it). A confirmed shock vetoes new entries and
remains latched until a completed candle covers the shock. The quote path
cannot publish directional snapshots or affect protective exits. No REST
pull, alternate classifier, or broker call is added.

The post-session report counts UNKNOWN/local RANGE overlap and denied minutes,
shadow transition candidates and clock-matched reference coverage. Fewer than
five prior observed sessions at the same minute is insufficient; no adaptive
threshold is promoted without chronological after-cost evidence.

Weekly NIFTY options (exchange numeric month codes 1–9 and O/N/D) and monthly
options are both accepted by the canonical syntax validator. This does not
infer expiry or authorize execution: the active basket, broker instrument row,
quote and original risk gates remain authoritative. Both canonical and
strategy-level denials reuse the same atomic basket observation and actual lot.
