# Runbook

## Rotate Zerodha tokens
1. `./manage_bot.sh token`
2. Follow the URL to log in and retrieve the `request_token`.
3. Paste the token when prompted; `.env` is updated with `ZERODHA__ACCESS_TOKEN`.
4. Restart the bot.

## Kill switch
- Send `/stop` via Telegram for an immediate halt.
- Or set `ENABLE_LIVE_TRADING=false` before launch to disable order placement.

## Restart procedure
1. Ensure environment variables are set in `.env`.
2. `./manage_bot.sh run`
3. Verify `curl localhost:8000/health` returns JSON.

## Common failure modes
- **Network down**: health check fails or no ticks. Verify connectivity and restart.
- **Broker reject**: order API returns an error. Review logs and `/apihealth` for circuit breaker status.
- **Stale ticks**: `/status` shows old timestamps. Restart data feed using the restart procedure.
