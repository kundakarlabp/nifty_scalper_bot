# Runbook

## Rotate Kite tokens
1. `./manage_bot.sh token`
2. Follow the URL to log in and retrieve the `request_token`.
3. Paste the token when prompted; `.env` is updated with `KITE_ACCESS_TOKEN`.
4. Restart the bot.

## Kill switch
- Live trading is on by default. Set `ENABLE_LIVE_TRADING=false` (or `ENABLE_TRADING=false`) before launch to avoid placing real orders.
- Send `/stop` via Telegram for an immediate halt.

## Restart procedure
1. Ensure environment variables are set in `.env`.
2. `./manage_bot.sh run`
3. Verify `curl localhost:8000/health` returns JSON.

## Daily execution checklist
- Before market open run `/execstate NIFTY` to confirm lifecycle levels and reconcile state.
- Run `/execqueue` to ensure there are no orphaned requests from the prior session.
- Trigger `/execlast` after the first trade to confirm fills are being recorded with expected latency.

## Post-deploy log verification
After each deployment, review recent logs and confirm:

1. `quote_diag` emits bid/ask details for a token, the subsequent `micro` line references the same token, and it does **not** report `reason=no_quote`.
2. `micro` log entries display a lot size of `75` for NIFTY.
3. The exposure `cap_pct` printed by both `gates` and `micro` logs matches exactly.
4. `resubscribe` messages for any token appear no more than once every 10–15 seconds.
5. There are no `UnboundLocalError` messages or crashes.
6. When trades are skipped, the log notes a confidence score below the configured threshold rather than `no_quote`.

## Common failure modes
- **Network down**: health check fails or no ticks. Verify connectivity and restart.
- **Broker reject**: order API returns an error. Review logs and `/apihealth` for circuit breaker status.
- **Stale ticks**: `/status` shows old timestamps. Restart data feed using the restart procedure.
- **No warm-up bars**: The bot requires market-hour ticks to build warm-up bars. When running with `DATA__WARMUP_DISABLE=true`, allow ~30 minutes of live ticks before trading. Use `/why` or `/force_eval` to diagnose or trigger evaluation.
- **Trades blocked by risk gates**: run `/execwhy SYMBOL` to view failing preflight gates and current metrics.

## Reference price & margin skips
- Inspect the `reference_price` log line; `source` should be one of `mid`, `ltp`, or `stale_mid` with a bounded `age_ms`. If `skip_reason=no_reference_price` appears, confirm market depth is available or relax `ORDER_REF_REQUIRE_DEPTH` or `ORDER_ALLOW_LTP_FALLBACK` in the environment.
- A `skip_reason=margin_no_qty` indicates the minimum lot breached capital limits. Review the `needed` versus `available` values in the log context and adjust buffers or requested size before retrying.

## Reference price ladder & reasons
- `reference_price` logs now reflect a ladder of `mid → ltp → stale_mid`. Quote ages are emitted as integers; anything above `ORDER_MAX_QUOTE_AGE_MS` means the feed is stale.
- When the ladder fails, orders short-circuit with `skip_order` reasons of `no_reference_price` or `margin_no_qty`. Check `metrics.strategy_skips_total{reason}` to monitor frequency and adjust the relevant env toggles if rates spike.
