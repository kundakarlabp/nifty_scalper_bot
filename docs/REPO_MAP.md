# Repository map

> Compact navigation map for ChatGPT, Copilot, and human reviewers. Paths under the runtime sections are relative to `src/nifty_scalper_bot/`.

## Fast start

1. Read `docs/AGENT_START_HERE.md`.
2. Use an exact error, event name, class, or function with `scripts/agent_context.py`.
3. Fetch only the ranked files and their direct callers/tests.
4. Read the full `AGENTS.md` before editing a high-risk runtime path.

For owner-created issues titled `[Agent Context] ...`, GitHub Actions automatically adds a ranked context report.

## Top-level layout

| Path | Purpose |
|---|---|
| `src/nifty_scalper_bot/` | Production bot code |
| `tests/` | Unit, architecture, integration, execution-safety, and deployment tests |
| `dashboard/` | Streamlit operations console |
| `deploy/`, `ops/` | AWS Lightsail and operational scripts |
| `scripts/` | Repository tooling, including agent context, validation, merge, and benchmark commands |
| `.agents/skills/` | Task-specific debugging, TDD, design, review, and worklog workflows |
| `docs/` | Architecture, operational, agent-reference, and machine-readable contract material |
| `benchmarks/agent/` | Historical regression benchmark manifest for coding-agent changes |
| `tests/fixtures/replay/` | Sanitized deterministic golden replay inputs |

## Architecture SSOT

Canonical ownership, runtime path, high-risk markers, validation routing, and architecture-lint rules live in:

```text
docs/architecture/agent_manifest.json
```

This repository map is navigational only. Do not copy ownership/routing tables back into this file; update the manifest and its tests instead.

## Support modules

### Contracts and symbols

- `instruments/active_contracts.py` — canonical symbol helpers and active NIFTY future resolution.
- `core/instrument_manager.py` — instrument dump, spot/future/options selection, symbol-token maps, ATM CE/PE basket.

### Market data and streaming

- `streaming/websocket_manager.py` — KiteTicker connection, callbacks, watchdog and reconnect behavior.
- `data/rest/zerodha_client.py` — low-level broker REST/WebSocket integration.
- `data/persistent_state.py` — persisted runtime state.
- `data/candle_engine.py` — candle construction and readiness.

### Strategy and market context

- `core/strategy_manager.py` — strategy scoring and allocation.
- `strategies/signal_generator.py` — scored signal production.
- `strategies/indicators.py` — indicator calculations.
- `core/market_regime.py` — market-regime detection and fan-out.

### Execution and risk

- `execution/order_manager.py` — canonical order-entry and submission facade.
- `execution/safe_order_manager.py` — live-mode/operator compatibility wrapper around OrderManager.
- `execution/bracket_manager.py` — canonical TP/SL/trailing and exit-lifecycle facade.
- `execution/readiness.py` — pure readiness/arming helpers.
- `execution/fill_ledger.py` — fill accounting and reconciliation where used.
- `risk/risk_manager.py` — risk limits and telemetry.

### Configuration and operations

- `config/settings.py` — runtime settings facade.
- `infra/metrics.py` — metrics.
- `dashboard/operations_console.py` — operator dashboard.
- `deploy/lightsail_release.sh` — staged Lightsail release path.

### Backtesting

- `backtesting/backtest_engine.py` — event-driven historical engine with simulated fills and costs.
- `backtesting/replay.py` — canonical historical replay harness over the live runtime pipeline.
- `backtesting/parity.py` — canonical live-vs-replay parity checks.
- `backtesting/premium_decay_backtest.py` — premium-decay strategy backtest harness.
- `backtest/` — legacy import-compatibility shims only; new internal code must use `backtesting/`.
- `tests/fixtures/replay/golden_market_path.csv` — canonical sanitized market-path replay fixture.
- `benchmarks/agent/historical_regressions.json` — historical engineering regression benchmark.
- `docs/contracts/contract_manifest.json` — machine-readable cross-module contract index.

## Source-to-test navigation

| Symptom or change | Start with | Focused tests |
|---|---|---|
| WebSocket timeout, reconnect, missing ticks | `streaming/websocket_manager.py`, MDM, broker client | `tests/streaming/`, `tests/data/` |
| Wrong symbol, expiry, ATM strike or token | InstrumentManager, active contracts | `tests/instruments/`, `tests/core/`, `tests/data/` |
| Missing OHLC or readiness blocked | MDM, candle engine, app readiness, runner | `tests/data/`, `tests/core/`, `tests/strategies/` |
| LTP-only quote, spread or depth issue | MDM, DataHub, quote models | `tests/data/`, execution/readiness tests |
| Duplicate signal or same-bar evaluation | runner, signal generator, candle identity | `tests/strategies/`, `tests/core/` |
| Risk/cooldown/capital blocker | risk manager, readiness, app | `tests/risk/`, `tests/core/` |
| Duplicate/rejected/partial order | order manager, safe manager, position manager, fill ledger | `tests/execution/`, canonical integration tests |
| SL/TP/trailing/restart issue | bracket manager, adaptive trailing, position/fill recovery | bracket and recovery tests under `tests/execution/` |
| Telegram spam or command problem | telegram controller, alert utilities | `tests/notifications/`, utility tests |
| Dashboard truth/export/rendering | dashboard modules | `tests/dashboard/` |
| Lightsail release/startup | deploy scripts, release guard | deployment and release-guard tests |

## Source-of-truth invariants

- Contract selection lives in InstrumentManager.
- MDM owns ticks, subscriptions, quote quality, and OHLC history.
- DataHub is read-only and owns no duplicate history.
- Readiness uses canonical app/MDM/runner/indicator state.
- OrderManager is the canonical live placement path.
- PositionManager owns position/pending state.
- BracketManager owns protective-exit state.
- Specific blocker reasons are required when trading is not ready.
- Paper, shadow, and live modes remain separate.

## Agent tooling

Use the single public façade:

```bash
python scripts/agent_task.py context --query "exact error or symbol"
python scripts/agent_task.py plan --files path/to/changed.py
python scripts/agent_task.py check --files path/to/changed.py
python scripts/agent_task.py full --files path/to/changed.py
```

The façade delegates to the existing focused tools, automatically selects relevant historical regression cases, and never starts the live trading runtime.

## Runtime pressure ownership map

- `data/market_data_manager.py`: owns bounded WebSocket tick ingress, protected-symbol ordering, low-priority coalescing/drop accounting, tick-pressure stats, and candle construction.
- `data/data_hub.py`: owns read-facade quote state plus bounded snapshot persistence for quotes/orders/positions; it must not select contracts or own history.
- `storage/hub_store.py`: owns JSON-safe conversion and SQLite snapshot storage.
- `main.py`: owns lightweight `/livez`, structured `/health/trading` and `/trading/status`, and event-loop lag reporting.
- `superlite_admin_core.py` and `dashboard/superlite_console.py`: own explicit admin/Streamlit status presentation and recovery from engine/admin timeouts.

Source-to-test navigation:

- Tick pressure: `tests/data/test_mdm_tick_coalescing.py`, `tests/test_mdm_event_loop_consumer.py`.
- Snapshot persistence: `tests/data/test_datahub_bounded_persistence.py`.
- Health/admin status: `tests/test_main_health_readiness.py`, `tests/dashboard/test_superlite_admin_core.py`.
- Deployment wiring: `tests/architecture/test_lightsail_release_contract.py`, `tests/dashboard/test_console_smoke.py`.
