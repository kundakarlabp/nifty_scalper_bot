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

## Historical Telegram and alerting incidents

These are preserved incident records, not instructions to enable live trading or bypass safeguards. The registered command implementation is the authority for current behavior.

### Telegram alert handler hygiene

The operator alert bridge is the direct source of aggregated alert messages. The Telegram package logger filter alone is not enough if the root alert handler receives a propagated third-party record. The alert bridge now drops only the generic PTB updater polling line and keeps classified internal Telegram events visible.

### Telegram polling error root cause

The repeated alert text

```text
telegram.ext.Updater: default_error_callback - Exception happened while polling for updates.
```

is emitted by python-telegram-bot's Updater default polling error callback. It is not an order or strategy failure. The bot can still send outbound heartbeat messages while inbound command polling is failing or transiently retrying.

## Root cause

`TelegramBot.start()` registered an Application error handler, but `Application.add_error_handler()` does not handle transport errors raised by `Updater.get_updates()` during polling. Because `updater.start_polling()` was called without an explicit `error_callback`, PTB used its own `default_error_callback`, which logs a generic ERROR without the exception type. The alert aggregator then grouped that third-party generic ERROR into repeated aggregated alerts.

This hid the real distinction between:

- transient Telegram network/timeout errors, and
- true duplicate-poller conflicts where another process/VM/laptop is polling the same bot token.

## Fix

- Add a synchronous Updater polling error callback.
- Pass it to `updater.start_polling(error_callback=...)`.
- Classify duplicate polling conflicts as `TELEGRAM_POLLING_CONFLICT`.
- Classify timeouts/network errors as `telegram_polling_transient` at WARNING level.
- Keep generic Telegram polling failures visible as `TELEGRAM_UPDATER_ERROR`.
- Add a host-wide token lock to prevent same-host duplicate pollers, in addition to the previous process-local guard.

### WebSocket pong alert storm

## Symptom

Telegram repeatedly emitted `websocket_pong_timeout` warnings with changing `age=` values and then sent one-event aggregation summaries for the same persistent condition.

## Root cause

The log-to-alert bridge keyed warning records by logger, level, and function. It forwarded every `Condition met:` record to the Telegram queue, so dynamic numeric fields made a persistent watchdog condition generate a new queued event on every logging cycle.

## Correction

`AlertLogHandler` now extracts the semantic condition name from `Condition met: <event>` records, includes that event in the alert key, and applies a five-minute monotonic repeat window before queue insertion.

The first occurrence remains visible, distinct conditions from the same function remain independent, ordinary warnings are not suppressed, and application logging plus WebSocket reconnect behavior are unchanged.
