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
