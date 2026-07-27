# Remaining live-path hardening plan

This draft PR is a planning shell only. Do not merge until the production and regression-test commits described below are added and CI is green.

## Required production corrections

1. Telegram notifier hardening
   - Require the attached runtime loop to be running and not closed.
   - Suppress both asyncio and concurrent-future cancellation in completion callbacks.
   - Close the dispatch coroutine if thread-safe submission races with loop shutdown.
   - Preserve immediate return on tick/order threads; never restore asyncio.run() or add per-alert threads.

2. Reconciliation lifecycle cleanup
   - Preserve last-known-good completion semantics from PR #929.
   - Guarantee active-run and in-progress cleanup for success, failure, and missing-position-manager paths.
   - Fail closed when position_manager is unavailable.
   - Keep startup fail-closed and staleness limits unchanged.

3. ATM-distance ranking
   - Preserve explicit valid distance metadata.
   - Otherwise derive abs(selected_strike - atm_strike) using the canonical strike parser.
   - Keep unknown/malformed data fail-closed at the existing sentinel.
   - Do not change ranking weights.

4. Minute-boundary candle grace
   - Add LIVE_BAR_CLOSE_GRACE_SECONDS, default 2 seconds and clamped to 0-5 seconds.
   - Apply only to expected closed-bar calculation.
   - Preserve future-timestamp and stale-bar enforcement.

5. Futures-context availability
   - Trace the exact producer before changing behavior.
   - Distinguish unavailable context (None + reason) from valid neutral zero.
   - Do not treat missing futures context as bullish, bearish, or neutral evidence.

## Required validation

- Focused notifier, reconciliation, runner-ranking, and live-safety tests.
- Live runtime simulation.
- Full pytest suite.
- No positions.json, databases, logs, coverage files, or temporary workflows in the final diff.
- No changes to strategy thresholds, stops, targets, sizing, broker APIs, risk limits, cooldowns, or signal arbitration.
