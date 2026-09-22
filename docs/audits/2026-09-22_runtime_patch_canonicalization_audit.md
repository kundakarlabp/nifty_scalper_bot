# Runtime patch canonicalization audit — 2026-09-22

## Purpose

Production install proof still reported runtime patching after the signal-pipeline cleanup. This audit reviewed the remaining import-time adapters and classified which ones are true duplicate owners versus larger hardening layers that need separate migrations.

## Live verification before code changes

Production build `44e1fb5f58ece579ce73876aaf9f45d008b11652` was confirmed live through the canonical AppDeploy/Supabase relay. Startup, data readiness, evaluation readiness, broker authentication, funds verification and capital capacity were healthy. The only active blocker at the time of verification was `market_closed`. No `ERROR`, `Traceback` or `Exception` lines were present in the checked log window.

The merged telemetry contract was visible live:
- `ENGINE_SUMMARY ... candidate_generated=... approved_candidates=...`
- Manager candidate and Runner final approval counters are separate.

## Confirmed duplicate runtime ownership

### 1. Polling failover

`core/app.py` already defined and called `_polling_failover_supervisor_iteration()`, but `core/polling_failover_runtime.py` replaced that function at import time. This created two implementations of one recovery path.

The deeper audit showed that the runtime adapter was compensating for a lower-level
freshness mismatch rather than fixing the right owner. `MarketDataManager.trading_feed_health()`
already publishes `required_symbol_recovery_active`, and the canonical
`decide_polling_fallback()` already treats that state as authoritative. However,
`MarketDataManager.classify_live_tick_readiness()` only checked packet-arrival
monotonic age. A stream of newly arriving packets carrying an old exchange/event
timestamp could therefore be classified as ready.

Canonical action in this PR:
- make `core/app.py` the only polling supervisor owner;
- preserve the deployed stale-futures compensation by consuming structured
  `readiness_blockers` (with the legacy primary-blocker string as fallback);
- preserve hysteresis, sync/async-safe fallback lifecycle, and change-based
  decision logging;
- delete the runtime replacement module;
- replace the patch-installed health marker with
  `polling_failover_native_owner=true` as the positive proof.

A deeper MDM market-event timestamp correction was prototyped but deliberately
removed from this PR. Deterministic live simulation uses a virtual market clock,
so a naïve wall-clock comparison falsely classified valid simulated ticks as
stale. The MDM file also carries unrelated legacy lint debt. Event-time
freshness therefore remains a separate owner-level task requiring an injected
clock/time-domain contract and dedicated parity tests.

### 2. Runner CandleEngine mirror cache

`StrategyRunner._mirror_authoritative_candle_engine()` is the natural owner of the MDM-owned engine mirror, but `core.__init__` wrapped the method at runtime only to return an already cached engine before reacquiring the MDM registry.

Canonical action:
- put the cache check directly in the Runner method;
- remove the installer/wrapper from `core.__init__`;
- preserve the MDM as the single authoritative CandleEngine owner.

## Deferred adapters

The following are real monkey-patch/adaptor mechanisms, but they touch broader runtime hot paths and are intentionally not bundled into this PR:

- `strategy_context_fast_path.py` wrapping `StrategyManager.generate_signal()`;
- `boot_readiness_safety.py` method replacements;
- `runtime_reliability_hardening.py` replacements across MDM/DataHub/Runner;
- off-market basket adapters;
- session-boundary rearm adapter;
- dynamic-universe and live-WS receipt adapters;
- the remaining `core.app` import hook that installs those adapters.

These should be migrated one owner at a time with dedicated regression tests. Removing the import hook before those migrations would create functional loss.

## Safety invariants preserved

This canonicalization does not change:
- strategy trigger thresholds;
- final alpha/quality thresholds;
- risk sizing or daily loss limits;
- broker/order routing;
- contract selection;
- SMC/VWAP/ORB entry rules;
- OrderFlow's context-only role.

## Validation contract

Regression coverage must prove:
1. the native polling supervisor preserves stale-futures recovery when the
   primary blocker names `futures_live_tick_stale`;
2. a higher-priority primary blocker cannot hide a structured
   `futures_live_tick_stale` recovery requirement;
3. unrelated blockers do not activate fallback on a healthy feed;
4. direct `core.app` import keeps the polling owner native;
5. repeated Runner engine resolution calls MDM only once after the mirror is cached;
6. runtime install proof reports native polling ownership and no polling runtime patch;
7. deterministic live simulation remains clock-domain compatible.

