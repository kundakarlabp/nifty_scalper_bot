# Nifty Scalper Bot – Deployment (Railway)

**Flags:** `ENABLE_LIVE_TRADING=true|false`, `BROKER_CONNECT_FOR_DATA=true` (connect broker for WS/data in paper mode), `DATA__WARMUP_DISABLE=true` (skip warmup).

## One-time
1. Add repository secrets or Railway variables for:
   - `KITE_API_KEY`, `KITE_ACCESS_TOKEN`
   - `TELEGRAM__BOT_TOKEN`, `TELEGRAM__CHAT_ID`
   - `ENABLE_LIVE_TRADING=true` (enable live trades; default is true; alias `ENABLE_TRADING`)
   - `DATA__TIME_FILTER_START=09:20`, `DATA__TIME_FILTER_END=15:25`

2. Commit `.env.example` for local runs.

## Procfile
This service runs:
```
worker: bash manage_bot.sh run
```

## Health
A background `http.server` listener exposes `GET /health` on
`0.0.0.0:$PORT` (defaults to `8080`).  Railway's health check path should be set
to `/health` so probes continue to work without bundling Flask/Waitress.

## Railway Variables
Set these in *Railway → Variables* (raw values without quotes):

- `ENABLE_LIVE_TRADING=false`
- `KITE_API_KEY=…`
- `KITE_API_SECRET=…`
- `KITE_ACCESS_TOKEN=…`
- `TELEGRAM__ENABLED=true`
- `TELEGRAM__BOT_TOKEN=…`
- `TELEGRAM__CHAT_ID=…`
- `DATA__TIME_FILTER_START=09:20`
- `DATA__TIME_FILTER_END=15:25`
- `HISTORICAL_TIMEFRAME=minute`
- `RISK__MAX_DAILY_DRAWDOWN_PCT=0.10`
- `EXPOSURE_CAP_PCT=2.0`
- `INSTRUMENTS__MIN_LOTS=1`
- `INSTRUMENTS__MAX_LOTS=5`
- `MAX_DATA_STALENESS_MS=30000`
- `WATCHDOG_STALE_MS=6000`
- `RECONNECT_DEBOUNCE_MS=20000`
- `STALE_X_N=3`
- `STALE_WINDOW_MS=60000`
- `MICRO__STALE_MS=1500`
- `ORDER_MAX_QUOTE_AGE_MS=6000`
- `QUOTES__MODE=FULL`

## Post-deploy verification

After Railway deploys a build:

1. Check Railway logs for `ws_resubscribe tokens=…` shortly after any reconnect
   and confirm there are no stack traces.
2. During market hours the decision loop should log `EVALUATE`/`decision` events;
   outside market hours the runner should remain in the `IDLE` phase.
3. In Telegram, run `/diag` to ensure `ws_connected=1` and the reported
   `last_tick_age_ms` looks reasonable, then run `/subs` to confirm FULL quotes
   are subscribed.

## Start/Stop jobs
Use the included scripts from a Railway Cron or separate services:
- `scripts/start_bot_morning.sh` (runs `/start` flow)
- `scripts/stop_bot_evening.sh` (idempotent `/stop`)

Both are safe to call multiple times.

## Graceful Shutdown
`manage_bot.sh` traps SIGTERM/SIGINT and forwards a clean stop to the trader before exiting.

## CI
A minimal GitHub Actions workflow runs lint/type-check/smoke-imports on every push to `main`.

---
Generated 2025-08-11 16:46:33 IST.