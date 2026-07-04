# Telegram operator console

This is the production Telegram command surface for the NIFTY scalper bot. The active registry lives in `src/nifty_scalper_bot/notifications/operator_telegram.py`; the older `telegram_commands.py` compatibility hook must not register commands.

## Daily commands

| Command | Purpose |
|---|---|
| `/status` | One-screen runtime state: mode, market, selected CE/PE, readiness and blocker. |
| `/why` | Exact reason the bot is not trading or why the latest candidate was rejected. |
| `/doctor` | One-shot triage combining status, blocker, execution state and top errors. |
| `/market` | Selected CE/PE quote, depth, tick age, bars and hydration state. |
| `/exec` | Execution blockers, open orders, positions, bracket attachment and unresolved exits. |
| `/risk` | Breaker, daily P&L, daily trade count and latest risk rejection. |
| `/positions` | Broker/local position comparison. |
| `/bracket` | Active virtual bracket, SL/TP/trailing and unresolved-exit state. |
| `/reconcile` | Read-only broker/local reconciliation report. |
| `/emergency` | Immediate kill switch / disable live damage path. |

## Diagnostics

| Command | Purpose |
|---|---|
| `/health` | System health and top blocker. |
| `/diag` | Broader diagnostics summary. |
| `/check` | Subsystem readiness overview. |
| `/check_connectivity` | Broker, WebSocket, DataHub and Telegram connectivity. |
| `/check_market` | Detailed market-data/hydration diagnostics. |
| `/check_core` | Strategy runner, regime and readiness diagnostics. |
| `/check_execution` | Live-order arming, risk, positions, open orders and bracket diagnostics. |
| `/today` | Today's P&L/trade count/open exposure summary. |
| `/latency` | Quote, WebSocket, REST and order latency snapshot. |
| `/version` | App version, build and git SHA. |
| `/selftest` | Non-invasive system self-test. |
| `/errors` | Recent error summary. |
| `/logs [N]` | Inline recent logs. |
| `/dumplogs [N]` | Download recent logs as a text file. |
| `/stderror` | Last captured runtime exception. |

## Control commands

| Command | Safety policy |
|---|---|
| `/pause` | Immediate; pauses new entries only and must not pause protective exits. |
| `/resume` | Requires `/confirm resume <code>`. |
| `/shadow` | Shows shadow mode when no argument is supplied. |
| `/shadow on` | Immediate safer action; enables shadow mode. |
| `/shadow off` | Requires `/confirm shadow_off <code>`. |
| `/flatten` | Requires `/confirm flatten <code>`. Intended for bot-owned positions only. |
| `/cancel_pending` | Requires `/confirm cancel_pending <code>`. Intended for non-protective/open orders. |
| `/confirm <action> <code>` | Confirms a pending sensitive action within the short confirmation window. |
| `/emergency` | Immediate emergency action; engages kill switch only. It is intentionally separate from flattening. |

## Design invariants

- One active registration path: `register_operator_commands()`.
- All diagnostic commands are read-only.
- `/emergency` is immediate and does not require confirmation.
- `/flatten`, `/cancel_pending`, `/resume`, and `/shadow off` are confirmed actions.
- `/pause` stops new entries but must not disable protective exit lifecycle management.
- The operator console does not place entry orders.
- Unauthorized chat IDs are rejected before any command handler is executed.
