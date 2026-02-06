# Nifty Scalper Bot

A modular, typed template for building intraday trading strategies for the Nifty index.

## Features

- Pydantic v2 configuration with environment loading
- Structured logging helpers
- Condition-based rate limiter with named buckets
- Dummy broker REST client with retry-aware throttling
- WebSocket manager scaffold with watchdog and reconnect logic
- Risk-aware order executor and simple RSI placeholder strategy
- TTL quote cache utilities and Telegram notification stub

## Getting Started

1. Create a virtual environment targeting Python 3.10 or newer.
2. Install dependencies:

   ```bash
   pip install -e .[dev]
   ```

3. Update the repository `.env` with your broker credentials. At minimum set:

   ```bash
   export BROKER_API_KEY=your_key
   export BROKER_API_SECRET=your_secret
   ```

4. Run the strategy loop locally using the dummy broker:

   ```bash
   python -m nifty_scalper_bot.app
   ```

## Tooling

- Formatting: [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/)
- Static typing: [mypy](https://mypy-lang.org/) with strict settings
- Testing: [pytest](https://pytest.org/)

Pre-commit hooks are available via `pre-commit install`.

## Operations

- Day-to-day runbook: see [`RUNBOOK.md`](RUNBOOK.md).
- Production hardening guide: see [`docs/production_playbook.md`](docs/production_playbook.md) for architecture, safety, observability, testing, and deployment checklists.

## Execution Monitoring

- `/execstate SYMBOL` — inspect lifecycle state, SL/TP levels, and unrealised PnL for a position.
- `/execqueue` — list queued `OrderRequest` objects with priority, source strategy, and age.
- `/execlast` — display the five most recent execution decisions with fill status and latency.
- `/execwhy SYMBOL` — surface preflight gates blocking a trade with current values and limits.
- `/emergency_stop` — pause the queue and send market exits for every open position.
- `/pause_trading` / `/resume_trading` — toggle queue intake while lifecycle management remains active.

## Railway Deployment Checklist (Zerodha WS)

**Required environment variables**

- `ZERODHA_API_KEY`
- `ZERODHA_ACCESS_TOKEN` (plain token)
- `ZERODHA_WS_ORIGIN=https://kite.zerodha.com`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

**Optional environment variables**

- `ZERODHA_WS_CONNECT_TIMEOUT=60`
- `ZERODHA_WS_RECONNECT_MAX_TRIES=50`

**Polling overrides**

- `POLL_INTERVAL_MS=700` (REST loop cadence)
- `POLL_BATCH_SIZE=200` (tokens per batch request)
- `POLL_REQUIRE_DEPTH=true` (force quote endpoint for depth data)
- `POLL_WARN_RATE_LIMIT=false` (silence high load warnings)

**Cloud deployment recommendation**

Polling mode is more reliable for Railway/Heroku/Cloud deploys; WebSocket/
webhook should be used only on static public IP/server with trusted domain and
TLS certificate. Recommended `.env` baseline:

```
WEBSOCKET__DISABLED=true
TELEGRAM__WEBHOOK_ENABLED=false
POLL_BATCH_SIZE=50
POLL_INTERVAL_MS_JITTER_PCT=0.15
```

**Enhanced infrastructure sample `.env` snippet**

```
ENABLE_LIVE=false
TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=replace-with-your-token
ALLOWED_CHAT_IDS=12345,67890
```

**Steps**

1. `git push` to trigger build.
2. Verify logs show “WebSocket connected” and “Flushing pending subscriptions”.
3. Run Telegram `/ws_status`, `/tick 256265`.
4. If 1006 appears, investigate:
   - Expired token → regenerate via login flow.
   - Concurrent sockets → stop other instances; one socket per token.

**REST auth verification**

```bash
curl -H "Authorization: token <API_KEY>:<ACCESS_TOKEN>" https://api.kite.trade/user/profile
```

## Testing

Run the lightweight test suite:

```bash
pytest
```

## Troubleshooting

- Orders not executing → run `/execwhy SYMBOL` to inspect preflight gates and current values.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `ENABLE_LIVE` | Toggle live order routing safeguards | `false` |
| `TELEGRAM_ENABLED` | Enable enhanced Telegram notifier when credentials exist | `true` when token provided |
| `TELEGRAM_TOKEN` | Telegram bot token override for enhanced notifier | required when `TELEGRAM_ENABLED=true` |
| `ALLOWED_CHAT_IDS` | Comma-separated Telegram chat IDs allowed to receive alerts | `` (inherit legacy chat) |
| `BROKER_API_KEY` | Broker API key | required |
| `BROKER_API_SECRET` | Broker API secret | required |
| `BROKER_ACCESS_TOKEN` | Optional session token | `None` |
| `BROKER_BASE_URL` | REST endpoint base | `https://api.example.com` |
| `BROKER_WS_URL` | WebSocket endpoint | `wss://ws.example.com/stream` |
| `BROKER_REST_CACHE_TTL_SEC` | Broker REST cache TTL for positions/margins fallback | `15.0` |
| `BROKER_MARGIN_SEGMENT` | Zerodha margin segment to inspect (`equity` or `commodity`) | `equity` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RISK_MAX_DAILY_TRADES` | Daily trade cap | `20` |
| `RISK_MAX_ORDER_NOTIONAL` | Max order notional | `200000.0` |
| `RISK_ALLOW_SHORT` | Allow short selling | `true` |
| `RISK_PER_TRADE_CAP_PCT` | Per-trade capital cap percentage for sizing | `1.0` |
| `RATE_LIMIT_ORDERS_CAPACITY` | Order bucket size | `5` |
| `RATE_LIMIT_ORDERS_REFILL_PER_SEC` | Order bucket refill rate | `5.0` |
| `RATE_LIMIT_REST_CAPACITY` | REST bucket size | `10` |
| `RATE_LIMIT_REST_REFILL_PER_SEC` | REST bucket refill | `10.0` |
| `RATE_LIMIT_HIST_CAPACITY` | Historical data bucket size | `2` |
| `RATE_LIMIT_HIST_REFILL_PER_SEC` | Historical bucket refill | `1.0` |
| `QUOTE_STALE_THRESHOLD_MS` | Quote staleness guard | `5000` |
| `POLL_INTERVAL_MS` | Polling interval for Zerodha REST streamer (ms) | `700` |
| `POLL_BATCH_SIZE` | Max tokens fetched per polling batch | `200` |
| `POLL_REQUIRE_DEPTH` | Force polling to use quote endpoint for depth | `false` |
| `POLL_WARN_RATE_LIMIT` | Emit warnings when polling load may exceed REST limits | `true` |
| `TICK_STALE_MS` | Drop ticks older than this age | `2000` |
| `EXECUTION_MODE` | Execution routing mode (`LIVE`, `PAPER`, `SHADOW`) | `SHADOW` |
| `EXECUTION_RETRY_ATTEMPTS` | Retry attempts for live orders | `3` |
| `EXECUTION_RETRY_DELAY_MS` | Delay between live order retries (ms) | `500` |
| `SHADOW_DRIFT_THRESHOLD_BPS` | Shadow slippage threshold before alerting | `20` |
| `SHADOW_DRIFT_AUTO_PAUSE` | Automatically pause live routing on repeated drift | `false` |
| `LIFECYCLE_TP1_R` | TP1 risk-reward multiple | `1.0` |
| `LIFECYCLE_TP1_PARTIAL` | Fraction of quantity exited at TP1 | `0.6` |
| `LIFECYCLE_TP2_R_TREND` | TP2 multiple when regime is TREND | `1.8` |
| `LIFECYCLE_TP2_R_RANGE` | TP2 multiple when regime is RANGE | `1.4` |
| `LIFECYCLE_TRAIL_ATR_MULT` | ATR multiple for trailing stop updates | `0.8` |
| `LIFECYCLE_TIME_STOP_MIN` | Time-based exit in minutes | `12` |
| `RECONCILIATION_INTERVAL_SEC` | Post-fill reconciliation interval | `30` |
| `RECONCILIATION_ALERT_ON_MISMATCH` | Emit alerts when drift is detected | `true` |
| `RECONCILIATION_BROKER_IS_TRUTH` | Treat broker snapshot as the source of truth | `true` |

### Live trading toggles

For integration tests that exercise the live order routing code paths, export the following environment variables to avoid `live trading disabled` guard rails:

```bash
export ENABLE_LIVE=1
export SESSION_ALLOW_OUT_OF_HOURS=1  # optional: bypass market hours check during testing
```

## License

MIT License.
