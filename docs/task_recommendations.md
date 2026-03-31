# Task Recommendations

This document captures follow-up tasks identified during the latest audit. Each entry includes the affected component and a concise fix proposal.

## Typo Fix
- **Issue**: The `TradeRecord` dataclass docstring in `strategies/runner.py` uses the British spelling "summarising," while the project standardises on American English elsewhere. 【F:src/nifty_scalper_bot/strategies/runner.py†L48-L56】
- **Proposal**: Update the docstring to say "summarizing" to keep terminology consistent across the codebase.

## Bug Fix
- **Issue**: `TelegramService.start` never clears `_stop_evt` before launching its polling loop. After `stop()` sets the event, any subsequent `start()` call exits immediately without restarting the worker. 【F:src/nifty_scalper_bot/notifications/telegram_service.py†L229-L258】
- **Proposal**: Clear `_stop_evt` at the beginning of `start()` so the service can be stopped and restarted without reconstructing the instance.

## Documentation Discrepancy
- **Issue**: `docs/REPO_MAP.md` links directly to `src/...` paths, but all modules live under `src/nifty_scalper_bot/...`, so the links 404 when rendered on GitHub. 【F:docs/REPO_MAP.md†L11-L39】
- **Proposal**: Update the Markdown links to include the `nifty_scalper_bot/` prefix so they point at the actual files.

## Test Improvement
- **Issue**: `tests/test_order_executor.py` only asserts that oversized notionals raise `OrderPlacementError`; it lacks a happy-path check that covers nonce reuse, price rounding, and open-order bookkeeping. 【F:tests/test_order_executor.py†L10-L24】
- **Proposal**: Add a complementary test that places a valid order and inspects the returned payload/open-order cache to guard against regressions in the execution path.
