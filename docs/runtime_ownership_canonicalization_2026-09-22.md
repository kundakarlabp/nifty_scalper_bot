# Runtime ownership canonicalization audit — 22 Sep 2026

## Scope

This audit followed production verification of main commit
`44e1fb5f58ece579ce73876aaf9f45d008b11652`. Production was healthy outside
market hours: startup/data/evaluation ready, broker funds verified, no current
ERROR/Traceback/Exception lines, and the expected `market_closed` execution
blocker.

The goal was to identify true runtime monkey patches that duplicate behavior
already owned by production classes/modules, then migrate only low-risk cases
into their canonical owners.

## Findings

### 1. Polling failover had two implementations

`core/app.py` already defined `_polling_failover_supervisor_iteration()`, while
`core/polling_failover_runtime.py` installed a second implementation over that
function during import. The runtime adapter contained newer safety behavior and
therefore acted as the real production owner despite `app.py` appearing to own
the path.

Canonical correction:
- move the deployed safety behavior into `core/app.py`;
- preserve structured `futures_live_tick_stale` recovery;
- preserve activation/recovery hysteresis;
- preserve sync/async-safe fallback start/stop;
- preserve change-based decision logging;
- remove the runtime replacement module;
- report native ownership in install proof instead of a patch-installed marker.

No polling thresholds or recovery criteria are loosened.

### 2. Runner CandleEngine cache was an import-time wrapper

`StrategyRunner._mirror_authoritative_candle_engine()` always called back into
MDM, while `core.__init__` wrapped the method at runtime to reuse the already
mirrored authoritative engine.

Canonical correction:
- perform the cache lookup directly in
  `StrategyRunner._mirror_authoritative_candle_engine()`;
- preserve MDM as the single CandleEngine owner;
- preserve first-use locking/identity behavior;
- remove the runtime method replacement;
- prove native method ownership in the runtime-hardening contract.

## Safety invariants

- No strategy score, alpha threshold, spread threshold, risk limit, daily loss
  limit, position sizing rule, order routing rule, or broker behavior is changed.
- Polling fallback remains a recovery path only.
- Futures stale readiness continues to force the existing REST recovery path
  even when packet-arrival freshness is misleading.
- Healthy feeds continue to suppress polling fallback.
- Runner never creates an independent CandleEngine; MDM remains authoritative.

## Deferred adapters

The audit also confirmed larger runtime adapters that remain non-canonical:
- `strategy_context_fast_path.py` replaces `StrategyManager.generate_signal`;
- boot-readiness safety replaces multiple app/Runner/MDM/Indicator methods;
- runtime-reliability hardening replaces MDM, DataHub and Runner hot-path
  methods;
- dynamic-universe, off-market and session-boundary adapters also mutate
  production classes/modules at runtime.

These are deliberately not bundled into this PR. They touch broader hot paths
and should be migrated owner-by-owner with dedicated parity tests.

## Validation required before merge

- normal pytest suite;
- deterministic broker-free E2E;
- compileall;
- Ruff;
- Black;
- Mypy;
- import safety/idempotency;
- polling fallback recovery tests;
- MDM/Runner CandleEngine identity/cache tests.

After deployment, production install proof should report
`polling_failover_native_owner=true` and should no longer report a polling
runtime patch as a required condition.
