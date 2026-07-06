# Market data hardening PRD/TDD

## Product requirement

The live market-data layer must have one deterministic WebSocket tick ingress path, strict freshness semantics for execution-critical symbols, resilient reconnect cleanup, and regression tests that prevent duplicate tick processing or stale-data promotion.

## Problem statement

The current implementation has several high-risk edge cases:

1. WebSocket batch ticks can be routed through both `MarketDataManager.process_ticks()` and a legacy per-tick callback when both are wired.
2. Missing or malformed tick timestamps can be normalized to current wall-clock time, making old data appear fresh.
3. WebSocket reconnect cleanup can raise from `ticker.close()` and interrupt recovery.
4. Candle closure depends on next-tick arrival unless a periodic flush path is added later.

## Scope in this PR

In scope:

- Prevent duplicate WebSocket tick enqueue when MDM batch ingress is present.
- Make ticker cleanup best-effort during reconnect handling.
- Tag normalized WebSocket ticks with timestamp quality.
- Reject synthetic/unknown timestamp quality from hard WebSocket LTP freshness.
- Add focused unit tests for the above.

Out of scope for this PR:

- Full refactor of `market_data_manager.py` into smaller modules.
- Replacing the fallback worker queue implementation.
- Wall-clock candle flush scheduler. This should be implemented as a follow-up because it touches lifecycle timing and readiness behavior.

## TDD acceptance tests

1. A WebSocket batch with one tick and an attached MDM must call `process_ticks()` once and must not call the legacy per-tick callback.
2. A WebSocket batch without an attached MDM must still use the legacy callback fallback.
3. `ticker.close()` errors must be suppressed and logged so reconnect cleanup survives.
4. A WebSocket tick with no broker/exchange timestamp must be marked `timestamp_quality=synthetic`.
5. `has_fresh_ws_ltp()` must not treat synthetic timestamp ticks as fresh.
6. A valid fresh WebSocket tick with exchange/broker timestamp must still pass freshness.

## Operational notes

This PR intentionally uses narrow hardening hooks rather than rewriting the 10k-line MDM file. The next larger cleanup should inline these changes after full local CI and live-paper validation.
