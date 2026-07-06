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
