# Production Playbook

This document captures a battle-tested set of practices for operating the Nifty Scalper bot in production. Use it as a companion to the runbook and deployment guides when hardening the system for live trading.

## 1. Architecture Hardening
- Keep a single orchestrator (`NiftyScalperApp`) that owns the lifecycle of background tasks, including the Telegram console.
- Defer WebSocket subscribe/mode/unsubscribe calls until `on_connect` and flush pending requests in bounded batches. Rotate Kite sessions safely.
- Add token-bucket rate limiters for every REST surface (orders, general REST, historical data) and surface current utilisation via `/status` or a dedicated `/ratelimit` command.
- Stamp market data with millisecond timestamps and reject stale data. Example guardrails: ticks older than 2s, LTPs older than 500ms.
- Default to "shadow" (paper) mode in production; flip to live trading only after smoke tests via `/shadow on|off`.

## 2. Safety Rails
- Circuit breaker conditions: broker 5xx bursts, stale quotes beyond threshold, repeated order rejects, and daily loss/drawdown breaches. Expose reasons via `/risk` and `/status`.
- Ensure idempotent order flows by attaching unique `client_order_id`s, deduplicating retries, and persisting the recent ID set for 24h.
- Provide kill switches: `/ws_reconnect`, `/shadow on`, and `/emergency` to cancel pending orders, flatten positions, and trip the breaker.
- Enforce execution limits by clamping order quantity using cash and notional caps, and validating instruments before placing orders.

## 3. Observability
- Emit structured JSON logs (fields: timestamp, level, module, event, symbol, quantity, price, order ID, WebSocket state, heartbeat age, latency, error). Retain a ring buffer for `/errors` and `/dumpLogs`.
- Track metrics: counters (orders submitted/filled/rejected, WS disconnects, REST retries, circuit trips), gauges (open positions, heartbeat age, queue depths), and histograms (REST latency, subscription latency, quote age). Expose via `/metrics` and optionally a Prometheus endpoint.
- `/status` should report WebSocket heartbeat delta, market-data health, tracked symbols, cache sizes, active strategies, open positions/orders, and breaker state.
- Allow on-demand profiling with `/profile <seconds>` while keeping it off by default.

## 4. Testing and CI
- Unit tests: WebSocket queue flushing and session refresh, strategy signal → order flow, risk rule enforcement.
- Integration tests: mocked Kite client for login/quote/order, Telegram command table/auth/HTML messages.
- Smoke tests (Railway deployment): `/ping` < 500ms, `/ws_status` heartbeat delta < 5s, `/quote NIFTY` returns LTP.
- Run `./run_checks.sh` locally; CI should fail fast on lint, type, unit, and backtest regressions.

## 5. Deployment (Railway)
- **Procfile**:
  ```
  web: PYTHONPATH=src uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port ${PORT:-8000}
  worker: ENABLE_EMBEDDED_HTTP_SERVER=false bash manage_bot.sh run
  ```
- **Environment variables**:
  - `APP_ENV=production`, `TZ=Asia/Kolkata`.
  - Zerodha: `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_ACCESS_TOKEN`.
  - Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_PUBLIC_BASE_URL` (or `TELEGRAM_WEBHOOK_ENABLED=false` with `TELEGRAM_ENABLE_POLLING_FALLBACK=true` for polling-only mode).
  - Rate limits: `ORDERS_CAPACITY=5`, `ORDERS_REFILL_PER_SEC=0.5`, `REST_CAPACITY=10`, `REST_REFILL_PER_SEC=2`, `HIST_CAPACITY=2`, `HIST_REFILL_PER_SEC=0.2`.
  - Freshness: `TICK_STALE_MS=2000`, `QUOTE_STALE_MS=800`.
  - Logging: `LOG_LEVEL=INFO`.
- Runtime: Python 3.10+, `python-telegram-bot>=21,<22`, `httpx`, `kiteconnect`. Keep the service stateless unless snapshots are required.
- Use HTTPX timeouts (3–5s connect/read) with two retries and jitter. WebSocket reconnect backoff: 1 → 30s with jitter.

## 6. Telegram Operations Console
- Core commands: `/opshelp`, `/status`, `/ws_status`, `/ws_reconnect`, `/mdm`, `/resolve SYMBOL`, `/tick SYMBOL`, `/quote SYMBOL`.
- Risk/trading: `/positions`, `/orders`, `/risk`, `/strategies`, `/shadow on|off`.
- Diagnostics: `/errors`, `/dumpLogs 400`, `/config`, `/env`, `/metrics ws`, `/ws_diag`, `/gc`, `/profile 5`, `/reload`.
- Restrict to a single chat ID and default to `/shadow on` before markets open.

## 7. WebSocket Strategy
- Mark connection status only in `on_connect`. Queue operations until connected and flush in batches of ~400 with 50ms spacing.
- On error/close: log code/reason, mark disconnected, backoff exponentially (1,2,4,…30s) and refresh sessions when tokens rotate.
- Track `last_heartbeat_monotonic()` and alert if the delta exceeds 5s (report via `/status`).

## 8. Data Integrity
- Warm the `InstrumentResolver` cache at boot (NSE instruments) and expose `/resolve`.
- Represent ticks with symbol, token, LTP, timestamp, bid/ask. Cache ~500 ticks per symbol with TTL; purge when markets close.

## 9. Security
- Mask secrets in logs and `/config` outputs.
- Lock Telegram commands to the configured chat ID.
- Pin dependencies (`requirements.txt`) and enable automated updates (Dependabot/Renovate).
- Use Railway masked variables and limit project access.

## 10. Runbooks
### Cold Start
1. Deploy from `main` and tail logs.
2. From Telegram run `/ping`, `/version`, `/whoami`.
3. `/status`: verify WebSocket heartbeat delta < 5s and strategies listed.
4. `/quote NIFTY` returns LTP.
5. Toggle `/shadow off` once validations pass.

### Market Hiccup
1. `/ws_status`: if heartbeat delta > 10s issue `/ws_reconnect` and capture `/ws_diag` for evidence.
2. Inspect `/errors`; use `/dumpLogs 500` for deeper investigations.
3. If quotes are stale, turn `/shadow on` and trip the breaker if required.

### Emergency
1. `/shadow on` to stop new trades.
2. Close positions via the position manager or `/emergency` to cancel, flatten, and trip the breaker.

## 11. Service Level Objectives
- Quote staleness p95 < 600ms during trading hours.
- WebSocket heartbeat gaps p95 < 3s; reconnect mean time to recovery < 30s.
- Order REST latency p95 < 1.2s.
- Zero duplicate client order IDs.
- No unhandled exceptions in the Telegram task.

## 12. Optional Shims
- `WebSocketManager` helpers:
  ```python
  def connection_state(self) -> str: ...
  def last_heartbeat_monotonic(self) -> float: ...
  def is_connected(self) -> bool: ...
  def reconnect(self) -> None: ...
  ```
- `MarketDataManager` helpers:
  ```python
  def pull_quote(self, symbol: str) -> dict: ...
  def get_latest_tick(self, symbol: str) -> dict | None: ...
  ```

Adopt these practices incrementally to keep the system safe, observable, and ready for production incidents.

## Broker authentication, balance, and readiness fail-closed behavior

- Zerodha terminal authentication failures (HTTP 401/403, token exceptions, invalid
  sessions, incorrect API key/access token, and authentication-related permission
  denials) are treated as fail-closed. The client latches the invalid-auth state,
  clears REST caches, raises `BrokerAuthenticationError`, and suppresses repeated
  authenticated REST calls until credentials are replaced by a controlled restart
  or a future explicit credential-refresh lifecycle.
- LIVE account funds are read only from `GET /user/margins/{segment}`. The legacy
  `GET /margins/{segment}` endpoint is not an account-balance fallback and must
  not be used for live risk capital.
- LIVE mode never substitutes `RISK_CAPITAL`, `RISK__CAPITAL`,
  `BACKTEST__CAPITAL`, cached simulation capital, default capital, or fabricated
  zero for an unavailable broker balance. Those variables are simulation inputs
  only and are ignored as live broker funds.
- Live execution arming requires valid broker authentication, a valid broker
  balance snapshot, and successful startup position reconciliation. Position
  reconciliation failures leave the process alive for diagnostics while keeping
  live orders unarmed.
- Health endpoints are separated by purpose:
  - `/livez`: process liveness for platform restarts; market closure and expired
    broker tokens do not make this endpoint fail.
  - `/readyz`: operational dependency readiness; broker auth or reconciliation
    failures return 503.
  - `/health/trading`: trading-domain status with blockers; returns a diagnostic
    payload even when trading is blocked.
- Closed-market conditions are expected, not subsystem failures. Off-hours basket
  retries should wait for the next useful market event instead of polling live
  depth repeatedly.
