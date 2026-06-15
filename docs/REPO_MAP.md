# Repository Map

> Accurate as of the current `main`. Paths are relative to `src/nifty_scalper_bot/`.
> For the non-negotiable runtime rules and the authoritative pipeline, see `AGENTS.md`.

## Top-Level Layout
- `src/nifty_scalper_bot/` – Production bot source (data, strategies, execution, risk, notifications, infra).
- `tests/` – Unit/integration tests mirroring `src/`. Note: many sync tests are no-ops under the
  current conftest hook — new/critical tests must be `async def` to actually execute.
- `deploy/`, `ops/`, `scripts/` – Deployment, monitoring, and local helper tooling.
- `docs/` – Reference documentation (this map, playbooks, prompts).

## Authoritative Runtime Path
```text
core/app.py
→ data/market_data_manager.py
→ data/data_hub.py
→ strategies/runner.py
→ execution/order_manager.py
→ execution/bracket_manager.py
→ notifications/telegram_controller.py
```
Do not bypass this flow with side-channel patches. Options are the only tradable
instrument; spot is direction/context only; futures is optional volume/context.

## Core Runtime Files (each carries a standardized module docstring)
- `core/app.py` – Orchestrator. Builds BotContext, wires every subsystem, owns the
  startup sequence and the readiness/arming SSOT.
- `data/market_data_manager.py` (MDM) – **Sole owner** of ticks and OHLC history;
  broker history fetch; tick fan-out via the message bus.
- `data/data_hub.py` – Read facade over MDM (quotes/OHLC/OI/context). Owns no history;
  selects no contracts.
- `strategies/runner.py` – Event-driven evaluation loop; applies gates; hands accepted
  signals to the order path; tracks runner/indicator history counts.
- `execution/order_manager.py` – THE live order path: placement, retries, idempotency,
  lifecycle against the broker.
- `execution/bracket_manager.py` – Virtual (internal) SL/TP, ATR trailing, multi-target
  exits, partial scaling, orphan resync.
- `notifications/telegram_controller.py` – Operator console: command handlers, single-chat
  auth, diagnostics, guarded controls, alerts.

## Key Support Modules
### Contracts & instruments
- `core/instrument_manager.py` – Authoritative contract selection from the instrument dump
  (futures/options/ATM). Token resolution.
- `instruments/active_contracts.py` – Canonical symbol helpers and active NIFTY future
  resolution from instruments.

### Execution support
- `execution/safe_order_manager.py` – Safety wrapper around OrderManager (guards/idempotency).
- `execution/position_manager.py` – Position and pending-order state of record.
- `execution/readiness.py` – Pure readiness/arming decision helpers used by the live gate.
- `execution/order_executor.py` – Separate non-live executor. **NOT** the live order path.

### Strategy support
- `core/strategy_manager.py` – Scores strategies and allocates between them.
- `strategies/signal_generator.py` – Produces scored signals from indicators.
- `strategies/indicators.py` – Technical indicator calculations.
- `core/market_regime.py` – Market-regime detection and fan-out.

### Data & streaming
- `data/rest/zerodha_client.py` – Low-level Kite REST + websocket client.
- `streaming/websocket_manager.py` – Hardened KiteTicker streaming.
- `data/persistent_state.py` – Persisted runtime state.

### Risk, config, infra
- `risk/risk_manager.py` – Risk guardrails and telemetry.
- `config/settings.py` – Runtime settings facade.
- `infra/metrics.py` – Metrics.

### Backtesting
- `backtesting/backtest_engine.py` – Event-driven historical replay through strategies
  with simulated fills/costs.

## SSOT Invariants (read before editing the data/readiness path)
- History is owned only by MDM. DataHub stores none; never gate readiness on DataHub bars.
- Readiness gates on mdm/runner/indicator bar counts via the canonical functions in
  `core/app.py` (`compute_history_readiness`, `compute_selected_option_history_readiness`).
- Contract selection lives in InstrumentManager; MDM/DataHub/runner consume, never select.
