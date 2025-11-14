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
