# AGENTS.md

## Overview
These instructions apply to the entire repository.
They are intended to keep the bot runnable today and easy to maintain in the future.

## Development workflow
- Python 3.11 is expected. Use the local `.venv` created by `run_checks.sh`.
- Source code lives in `src/` and tests in `tests/`. Keep this structure intact.
- Favor small, typed functions and dataclasses. Include docstrings for public APIs.
- Run `./run_checks.sh` before every commit. This script installs deps, runs Ruff, mypy, pytest and a lightweight backtest.
- Resolve all Ruff findings (lint, style) and mypy issues. Unused imports and variables should be removed.
- Add or update tests when fixing bugs or adding features.
- Avoid hard‑coding secrets or credentials. Use `src.config.settings` and environment variables.

## Commit guidelines
- Use clear, present‑tense commit messages (e.g., `fix: remove unused import`).
- Group related changes into a single commit. Do not amend or rebase published history.
- Update README or in‑code comments when behavior or configuration options change.

## Pull requests
- Ensure `./run_checks.sh` succeeds and include the command output in the PR description.
- Summarize the problem, the approach, and testing performed.

## Order Execution Hub Architecture
- `StateTracker` persists order and position lifecycle data in SQLite and must be the source of truth for execution state.
- `PreFlightValidator` owns all gating checks before an order is released to brokers; wire new checks through this module.
- `LifecycleManager` automates stop-loss, take-profit, and trailing logic and records lifecycle events through `StateTracker`.
- `OrderQueue` prioritises intake, enforces deduplication, and should be the only entry point for new `OrderRequest` objects.
- `ExecutionRouter` dispatches requests based on mode (LIVE/SHADOW/PAPER) and tracks drift metrics in shadow mode.
- `PostFillMonitor` reconciles broker and local state and must run on startup to align open positions.

