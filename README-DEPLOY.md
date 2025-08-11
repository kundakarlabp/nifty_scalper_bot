# Nifty Scalper Bot – Deployment (Railway)

## One-time
1. Add repository secrets or Railway variables for:
   - `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN`
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - Optional: `ENABLE_LIVE_TRADING=false` (start in shadow mode)
   - `TIME_FILTER_START=09:15`, `TIME_FILTER_END=15:30`

2. Commit `.env.example` for local runs.

## Procfile
This service runs:
```
worker: bash manage_bot.sh run
```

## Health
A lightweight Flask server exposes `GET /health` on port 8000 to help Railway detect unhealthy states and restart.

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