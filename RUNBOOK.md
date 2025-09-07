# Runbook

## Rotate Zerodha tokens
1. `./manage_bot.sh token`
2. Follow the URL to log in and retrieve the `request_token`.
3. Paste the token when prompted; `.env` is updated with `ZERODHA__ACCESS_TOKEN`.
4. Restart the bot.

## Kill switch
- Live trading is off by default. Set `ENABLE_LIVE_TRADING=true` (or `ENABLE_TRADING=true`) before launch to place real orders.
- Send `/stop` via Telegram for an immediate halt.

## Restart procedure
1. Ensure environment variables are set in `.env`.
2. `./manage_bot.sh run`
3. Verify `curl localhost:8000/health` returns JSON.

## Common failure modes
- **Network down**: health check fails or no ticks. Verify connectivity and restart.
- **Broker reject**: order API returns an error. Review logs and `/apihealth` for circuit breaker status.
- **Stale ticks**: `/status` shows old timestamps. Restart data feed using the restart procedure.
