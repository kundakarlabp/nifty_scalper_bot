# Market data hardening PRD/TDD

## Product requirement

The live market-data layer must have one deterministic WebSocket tick ingress path, strict freshness semantics for execution-critical symbols, resilient reconnect cleanup, thread-safe fallback ingestion, and deterministic one-minute candle closure even when the next tick is delayed.

## Problem statement

The implementation had several high-risk edge cases:

1. WebSocket batch ticks could be routed through both `MarketDataManager.process_ticks()` and a legacy per-tick callback when both were wired.
2. Missing or malformed tick timestamps could be normalized to current wall-clock time, making old data appear fresh.
3. WebSocket reconnect cleanup could raise from `ticker.close()` and interrupt recovery.
4. Candle closure depended on next-tick arrival.
5. The fallback worker path used an asyncio queue from a synchronous/threaded callback context.
6. WebSocket trading-window checks ignored the configured timezone object.

## Completed scope

Implemented:

- Prevent duplicate WebSocket tick enqueue when MDM batch ingress is present.
- Make ticker cleanup best-effort during reconnect handling.
- Tag normalized WebSocket ticks with timestamp quality.
- Reject synthetic/unknown timestamp quality from hard WebSocket LTP freshness.
- Replace the no-loop fallback ingestion path with a `queue.Queue` backed worker.
- Add a clock-based candle flush task started with the MDM event-loop consumer.
- Use the configured WebSocket trading timezone for trading-window checks.
- Add focused unit tests for the above behavior.

Still out of scope:

- Full refactor of `market_data_manager.py` into smaller modules.
- Inlining all hardening hooks back into the primary source files. The hooks are intentionally narrow to reduce regression risk.
- Live broker validation. This must be done with paper/live-small mode during market hours.

## TDD acceptance tests

1. A WebSocket batch with one tick and an attached MDM must call `process_ticks()` once and must not call the legacy per-tick callback.
2. A WebSocket batch without an attached MDM must still use the legacy callback fallback.
3. `ticker.close()` errors must be suppressed and logged so reconnect cleanup survives.
4. A WebSocket tick with no broker/exchange timestamp must be marked `timestamp_quality=synthetic`.
5. `has_fresh_ws_ltp()` must not treat synthetic timestamp ticks as fresh.
6. A valid fresh WebSocket tick with exchange/broker timestamp must still pass freshness.
7. No-loop WS ingestion must use the thread-safe fallback queue, not the asyncio queue.
8. The fallback queue must coalesce/drop lower-priority work before protected open-position ticks.
9. Idle candles must finalize after the minute closes plus grace time, without requiring the next tick.
10. WebSocket trading-window logic must use the configured timezone.

## Operational notes

The runtime remains designed for shadow/paper validation before increasing live size. The safest next larger cleanup is to inline these hooks into the source modules after local CI and one full market-session paper run.

## Historical market-data validation and timestamp incident

These are preserved historical test checklists and incident details; current code and CI determine present behavior.

### Market data completion validation

The market-data manager classifies stale required symbols; the polling decision
helper evaluates that health state. `core/app.py` owns the polling supervisor,
including activation and recovery cooldown. Polling remains a recovery path and
must not replace fresher WebSocket FULL quotes.

Run these checks locally before increasing live size:

```bash
python -m compileall -q src tests
python -m pytest -q \
  tests/streaming/test_market_data_hardening.py \
  tests/data/test_market_data_hardening.py
python -m pytest -q
```

Market-hours paper/live-small validation checklist:

1. Confirm one `FIRST_TICK_RECEIVED` event per WS session and no duplicate-enqueue warnings.
2. Confirm `MDM_CANDLE_FLUSH_TASK_STARTED` after the MDM event loop is wired.
3. Watch for `MDM_CANDLE_CLOCK_FLUSHED` after idle one-minute buckets.
4. Confirm stale/synthetic WS LTP does not arm hard trading readiness.
5. Confirm fallback worker log appears only before the main asyncio loop is wired.
6. Keep live order size minimal until one complete market session passes without feed-health degradation.

### Future candle timestamp incident

Root cause: a naive internal IST wall-clock candle could be interpreted as UTC and become a later stored candle. Valid current bars were then rejected as out-of-order.

Fix:
- localize naive market-data timestamps as Asia/Kolkata
- convert aware timestamps to Asia/Kolkata
- reject ahead-of-clock candles before store insertion
- move overlap tolerance into CandleStore

Validation:
- python -m compileall -q src tests
- python -m pytest -q tests/data/test_future_candle_guard.py tests/data/test_volume_zero_preservation.py tests/data/test_market_data_pipeline.py tests/data/test_pipeline_overlap_guard.py tests/data/test_candle_engine.py
