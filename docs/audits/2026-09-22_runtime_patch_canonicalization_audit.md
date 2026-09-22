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

The runtime adapter contained newer safety behavior that must not be lost:
- structured `futures_live_tick_stale` blocker forces required-symbol recovery;
- async/sync-safe fallback start/stop handling;
- anti-flap recovery cooldown;
- fail-closed non-fatal supervisor behavior;
- decision observability.

Canonical action:
- preserve the newer safety behavior in `core/app.py`;
- delete the runtime replacement module;
- keep `polling_failover_runtime_patch_installed=false` as a compatibility observability field;
- add `polling_failover_native=true` as the positive proof.

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
1. stale futures readiness still activates the existing REST recovery path;
2. unrelated blockers do not activate fallback on a healthy feed;
3. market-closed fallback is stopped safely;
4. direct `core.app` import keeps the polling owner native;
5. repeated Runner engine resolution calls MDM only once after the mirror is cached;
6. runtime install proof reports native polling ownership and no polling runtime patch.

