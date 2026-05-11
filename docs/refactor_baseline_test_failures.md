# Baseline test failures (pre-refactor)

## Commands run
- `python -m compileall src`
- `pytest -q tests`

## Failures observed
1. `tests/test_mdm_diagnostics.py` import error:
   - `ModuleNotFoundError: No module named 'market_data_manager'`
