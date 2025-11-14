# Telegram Bot Setup Guide

This guide explains how to enable the production-grade Telegram bot controller that now ships with the project. The bot uses `python-telegram-bot` v21 and integrates with the existing configuration models, strategy runner, and broker abstractions.

## 1. Install Dependencies

The project now depends on:

- `python-telegram-bot>=21.0`
- `requests>=2.31`

Install them alongside the existing requirements:

```bash
pip install -e .
```

Running `./run_checks.sh` will also install the dependencies into the managed virtual environment.

## 2. Environment Configuration

Set the following variables in your `.env.local` (or environment of choice):

```dotenv
TELEGRAM__BOT_TOKEN=123456:ABCDEF
TELEGRAM__CHAT_ID=11111111   # numeric chat id for the single authorized user
ALLOW_LIVE_ORDERS=false      # keep false for dry-run testing
```

`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` remain supported for backward compatibility, but new deployments should prefer the double-underscore variants above. Legacy environment names such as `TELEGRAM_ALLOW_POLL_FALLBACK` continue to function, although the updated `TELEGRAM_ENABLE_POLLING_FALLBACK` flag matches the runtime configuration logs.

### Webhook deployments (FastAPI + Railway)

For webhook-first hosting platforms, expose the FastAPI app and register the webhook on startup:

```dotenv
TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=123456:ABCDEF         # rotated bot token
TELEGRAM_PUBLIC_BASE_URL=https://your-app.up.railway.app
TELEGRAM_ENABLE_POLLING_FALLBACK=false  # flip to true only when debugging locally
# Optional override: TELEGRAM_WEBHOOK_ENABLED=false to run in polling-only mode
TELEGRAM_ALLOWED_IDS=6931456598      # optional, comma separated chat IDs
```

Railway (and other ASGI hosts) expect an `app` object, which now lives at
`nifty_scalper_bot.main:app`. When the process boots, the webhook is registered
automatically; if it fails and fallback polling is disabled you will see a
`telegram_webhook_not_configured` warning in the logs.

## 3. Running the Bot

Start the bot with:

```bash
python -m nifty_scalper_bot.app
```

When Telegram credentials are missing, the strategy runner continues to operate and the process waits for `SIGINT`/`SIGTERM`. When credentials are supplied, the Telegram bot launches alongside the runner in the same process.

### Signals and Shutdown

- `Ctrl+C` (SIGINT) shuts down the Telegram bot and leaves the process gracefully.
- `SIGTERM` triggers the same graceful shutdown path.

## 4. Available Commands

| Command | Description |
| --- | --- |
| `/help` | List available commands. |
| `/ping` | Measure round-trip latency. |
| `/status` | Display runner, stale-threshold and metrics snapshot. |
| `/quote <SYMBOL>` | Fetch the latest quote via the throttled broker client. |
| `/order <BUY|SELL> <SYMBOL> <QTY> [PRICE]` | Place a live order (disabled unless `ALLOW_LIVE_ORDERS=true`). |
| `/risk` | Show the current risk configuration. |
| `/ratelimit` | Inspect rate limiter buckets. |
| `/ws [reconnect]` | Show websocket status or trigger a reconnect. |
| `/diag <topic>` | Run diagnostics (`imports`, `files`, `config`, `deps`, `network`, `log`, `errors`). |
| `/test <case>` | Execute component tests (`freshness`, `retry`, `rate`, `cache`, `strategy`, `executor`). |
| `/admin <cmd>` | Administrative commands (`set_log`, `reload_config`). |

### Safety

- Only the configured `TELEGRAM__CHAT_ID` can interact with the bot; messages from any other chat are ignored with a warning log.
- Live orders are blocked unless `ALLOW_LIVE_ORDERS=true`.
- `/admin reload_config` prevents on-the-fly changes to broker credentials.

## 5. Diagnostics and Logging

- A thread-safe ring buffer retains recent log lines for `/diag log` and `/diag errors`.
- Command metrics track total usage and error counts per command.
- DNS checks validate broker and webhook hostnames via `/diag network`.
- File system scans are cached for 60 seconds to keep `/diag files` responsive.

## 6. Testing Utilities

The `/test` command family enables smoke-testing core components:

- **freshness**: Validate `FreshnessGuard` behaviour for a supplied age.
- **retry**: Exercise retry logic by monkey-patching the broker client.
- **rate**: Run rate-limiter acquisitions to surface throttling.
- **cache**: Inspect cache expiry semantics using `TTLCache`.
- **strategy**: Execute a single strategy step with a dry-run executor.
- **executor**: Attempt to place an order using the live or dry-run executor.

## 7. Verification Script

Use `verify_installation.py` (see repo root) to confirm dependencies, environment variables and module imports before deploying. The script exits non-zero if required checks fail, making it suitable for CI/CD hooks.

## 8. Troubleshooting

- **Unauthorized**: Double-check that `TELEGRAM__CHAT_ID` matches your numeric Telegram chat identifier.
- **Rate limited**: Wait a minute; commands are limited per chat to avoid spam.
- **Broker unavailable**: Confirm broker credentials are loaded and the throttled client can connect.
- **Webhook diagnostics**: Use `/diag network` to confirm DNS resolution for broker and webhook endpoints.

## 9. Going Live

Flip `ALLOW_LIVE_ORDERS=true` only after verifying the bot in dry-run mode. Monitor logs through `/diag log` immediately after enabling live trading.

Happy trading! 🚀
