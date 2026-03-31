# Follow-up Task Proposals

## Typo Fix
- **Issue**: `docs/deployment/README.md` uses the British spelling "summarises," which is inconsistent with the rest of the repository's American English style guides.
- **Proposed Task**: Replace "summarises" with "summarizes" in that README so the documentation follows a single dialect.

## Bug Fix
- **Issue**: `scripts/true_backtest_dynamic.py` imports modules via the legacy `src.` namespace, e.g., `from src.boot.validate_env import ...`, but the actual package lives under `nifty_scalper_bot`. Running the script raises `ModuleNotFoundError`.
- **Proposed Task**: Update the script to import from the correct `nifty_scalper_bot` modules (or adjust `PYTHONPATH`) so it can run without manual path tweaks.

## Documentation Discrepancy
- **Issue**: `docs/REPO_MAP.md` links to paths such as `src/notifications/telegram_controller.py`, but all production code is nested under `src/nifty_scalper_bot/`. The links 404 when clicked.
- **Proposed Task**: Correct the repo map links to include the `nifty_scalper_bot` prefix so they point to real files.

## Test Improvement
- **Issue**: `tests/test_freshness_guard.py` only verifies that stale quotes raise `DataStaleError`; it never asserts that fresh quotes pass or that invalid payloads raise errors.
- **Proposed Task**: Expand the test to cover a fresh quote success path and malformed input handling to prevent regressions in the guard logic.
