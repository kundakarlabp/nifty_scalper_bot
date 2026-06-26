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
| `scripts/` | Repository tooling, including agent context and validation planning |
| `.agents/skills/` | Task-specific debugging, TDD, design, review, and worklog workflows |
| `docs/` | Architecture, operational, and agent-reference material |

## Authoritative runtime path

```text
core/app.py
→ data/market_data_manager.py
→ data/data_hub.py
→ strategies/runner.py
→ execution/order_manager.py
→ execution/bracket_manager.py
→ notifications/telegram_controller.py
```

Do not bypass this flow. Options are the only tradable instruments. Spot is direction/context only; futures is optional context only.

## Core ownership

| File | Owns | Does not own |
|---|---|---|
| `core/app.py` | Wiring, startup sequence, readiness/arming source of truth | Contract selection, history storage, order placement |
| `core/instrument_manager.py` | Contract selection and token resolution | Tick/history storage, strategy decisions |
| `data/market_data_manager.py` | Tick cache, subscriptions, quotes/depth/OI, OHLC history and hydration | Contract selection, strategy logic |
| `data/data_hub.py` | Read facade over active market data | Independent history or contract selection |
| `data/candle_engine.py` | Tick-to-OHLC bars and bar readiness | Broker instruments |
| `strategies/runner.py` | Evaluation loop, gates, signal-to-order handoff | Contract selection, broker history fetch, order placement |
| `execution/order_manager.py` | Canonical live placement, retries, idempotency, order lifecycle | Signal generation and contract selection |
| `execution/position_manager.py` | Position and pending-order state | Strategy scoring |
| `execution/bracket_manager.py` | Virtual SL/TP, trailing, targets, partial exits and recovery | Entry decisions and separate placement path |
| `notifications/telegram_controller.py` | Operator commands, authentication, diagnostics and alerts | Trading decisions and direct order ownership |

`execution/order_executor.py` is non-live and is not the canonical live order path.

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

- `execution/safe_order_manager.py` — safety wrapper around OrderManager.
- `execution/readiness.py` — pure readiness/arming helpers.
- `execution/lifecycle_manager.py` — lifecycle coordination where used.
- `execution/fill_ledger.py` — fill accounting and reconciliation where used.
- `risk/risk_manager.py` — risk limits and telemetry.

### Configuration and operations

- `config/settings.py` — runtime settings facade.
- `infra/metrics.py` — metrics.
- `dashboard/operations_console.py` — operator dashboard.
- `deploy/lightsail_release.sh` — staged Lightsail release path.

### Backtesting

- `backtesting/backtest_engine.py` — event-driven historical replay with simulated fills and costs.

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

## High-risk paths

```text
src/nifty_scalper_bot/core/app.py
src/nifty_scalper_bot/core/instrument_manager.py
src/nifty_scalper_bot/data/market_data_manager.py
src/nifty_scalper_bot/data/data_hub.py
src/nifty_scalper_bot/streaming/websocket_manager.py
src/nifty_scalper_bot/strategies/runner.py
src/nifty_scalper_bot/risk/risk_manager.py
src/nifty_scalper_bot/execution/order_manager.py
src/nifty_scalper_bot/execution/position_manager.py
src/nifty_scalper_bot/execution/bracket_manager.py
src/nifty_scalper_bot/notifications/telegram_controller.py
```

Inspect direct call sites, state ownership, restart/reconnect behavior, and regression tests before editing these files.

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

Generate ranked repository context:

```bash
python scripts/agent_context.py --query "exact error or symbol" --output /tmp/agent-context.md
```

Generate a focused validation plan:

```bash
python scripts/agent_check.py --files path/to/changed.py --output /tmp/agent-check.md
```

Final validation remains:

```bash
python -m compileall -q src dashboard
python -m pytest -q
```
