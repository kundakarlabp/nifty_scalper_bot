# Nifty Scalper Bot – Deployment (Railway)

**Flags:** `ENABLE_LIVE_TRADING=true|false`, `BROKER_CONNECT_FOR_DATA=true` (connect broker for WS/data in paper mode), `DATA__WARMUP_DISABLE=true` (skip warmup).

## One-time
1. Add repository secrets or Railway variables for:
   - `KITE_API_KEY`, `KITE_ACCESS_TOKEN`
   - `TELEGRAM__BOT_TOKEN`, `TELEGRAM__CHAT_ID`
   - `TELEGRAM_PUBLIC_BASE_URL=https://<your-app>.up.railway.app`
   - `ENABLE_LIVE_TRADING=true` (enable live trades; default is true; alias `ENABLE_TRADING`)
   - `DATA__TIME_FILTER_START=09:20`, `DATA__TIME_FILTER_END=15:25`

   Keep Railway focused on credentials only. Place non-secret toggles such as `STREAMING_MODE=polling`, `POLL_INTERVAL_MS=700`, and `STRATEGIES=EMA_CROSS: NIFTY` inside the repository `.env` so they can be versioned alongside the code. When running without a public ingress you can opt into polling-only Telegram delivery via:

   ```dotenv
   TELEGRAM_ENABLE_POLLING_FALLBACK=true
   TELEGRAM_WEBHOOK_ENABLED=false
   ```

2. Commit the populated `.env` for local runs.

## Procfile
Railway now launches dedicated processes for inbound webhooks and the trading
worker:

```
web: PYTHONPATH=src uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: ENABLE_EMBEDDED_HTTP_SERVER=false bash manage_bot.sh run
```

The `web` process exposes `/telegram/webhook` publicly, while the worker keeps
the trading stack isolated and disables the embedded HTTP server via
`ENABLE_EMBEDDED_HTTP_SERVER=false`.

## Health
The health endpoint is served by the FastAPI application on port 8000, ensuring
Railway can detect unhealthy states and restart the service automatically.

## Start/Stop jobs
Use the included scripts from a Railway Cron or separate services:
- `scripts/start_bot_morning.sh` (runs `/start` flow)
- `scripts/stop_bot_evening.sh` (idempotent `/stop`)

Both are safe to call multiple times.

## Graceful Shutdown
`manage_bot.sh` traps SIGTERM/SIGINT and forwards a clean stop to the trader before exiting.

## CI
A minimal GitHub Actions workflow runs lint/type-check/smoke-imports on every push to `main`.

## Troubleshooting: temporarily relax order gates
High-latency Railway regions can occasionally trigger stale quote or microstructure
blocks before an order is attempted. To confirm whether the guardrails are causing
the rejection, redeploy once with the following relaxed overrides:

```
ALLOW_NO_DEPTH=true
ORDER_MAX_QUOTE_AGE_MS=60000
ORDER_MAX_SPREAD_PCT=0.009
MICRO_MIN_OK_STREAK=1
MICRO_GRACE_MS=5000
WATCHDOG_STALE_MS=15000
```

If `order.attempt` logs begin to appear you have isolated the issue to gating. Revert
the overrides and tighten them back to normal once testing is complete.

---
Generated 2025-08-11 16:46:33 IST.