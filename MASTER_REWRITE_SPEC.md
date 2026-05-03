# NiftyScalperBot v2 — Master Rewrite Specification

**Document Version:** 1.0.0  
**Date:** 2026-04-02  
**Status:** AUTHORITATIVE — Single Source of Truth for Full Rewrite  
**Scope:** Complete production-grade Python trading bot for Nifty intraday options scalping

---

> **DEPRECATED SPEC:** Do not implement OrderExecutionHub, ExecutionRouter, or PreflightValidator. Current runtime path is StrategyRunner → OrderManager → BracketManager.

> **Reading Guide for AI Implementers:** Read this document top to bottom before writing a single line of code. Every section builds on the previous. Section 7 (per-file specs) is the implementation contract. Section 8 (strategy specs) contains the exact trading logic. If any detail seems ambiguous, the intent is always: capital preservation first, consistent alpha second, clean code third.

---

## TABLE OF CONTENTS

1. [Project Overview](#section-1-project-overview)
2. [Technology Stack](#section-2-technology-stack)
3. [Complete Directory Structure](#section-3-complete-directory-structure)
4. [Configuration Spec](#section-4-configuration-spec)
5. [Data Models](#section-5-data-models)
6. [Core Architecture and Flow](#section-6-core-architecture-and-flow)
7. [Per-File Implementation Specs](#section-7-per-file-implementation-specs)
8. [All 12 Strategy Implementations](#section-8-all-12-strategy-implementations)
9. [Risk Management Spec](#section-9-risk-management-spec)
10. [Telegram Commands Spec](#section-10-telegram-commands-spec)
11. [Database Schema](#section-11-database-schema)
12. [Testing Strategy](#section-12-testing-strategy)
13. [Improvements Over Existing Bot](#section-13-improvements-over-existing-bot)
14. [Deployment Spec](#section-14-deployment-spec)
15. [Known Limitations and Out of Scope](#section-15-known-limitations-and-out-of-scope)

---

## SECTION 1: PROJECT OVERVIEW

### 1.1 Identity

| Field | Value |
|-------|-------|
| **Bot Name** | NiftyScalperBot v2 |
| **Version** | 2.0.0 |
| **Language** | Python 3.11+ |
| **Market** | NSE Nifty 50 Index Options (Weekly + Monthly expiry) |
| **Broker** | Zerodha Kite (primary); BrokerGateway ABC for future multi-broker |
| **Trading Style** | Intraday options scalping (long calls/puts only, never naked short) |
| **Session** | 09:20 IST – 15:15 IST (avoids first 5 min chaos and last 15 min spread blow-out) |

### 1.2 Mission Statement

Build an **autonomous, fault-tolerant, observable** intraday options scalping bot for the Nifty index that:

1. **Preserves capital above all else** — hard daily loss limits enforced in code, not just policy
2. **Generates consistent alpha** — 12 complementary strategies that work in different market regimes
3. **Fails gracefully** — every external call has retry logic, timeouts, and circuit breakers
4. **Is fully observable** — Prometheus metrics, Grafana dashboards, Telegram operator console
5. **Is maintainable** — clean module boundaries, no file over ~1,000 lines, comprehensive tests

### 1.3 Architecture Philosophy

- **Modular**: Each file has exactly one responsibility. Import graph is a DAG, never cyclic.
- **Async-first**: All I/O (broker, DB, Telegram) is async. CPU-bound work (indicators) is synchronous numpy/pandas.
- **Event-driven**: Ticks flow through a fan-out EventBus. Strategies subscribe; they never poll.
- **Observable**: Every state transition emits a metric and a log line with structured fields.
- **Testable**: Every module is injectable. No global mutable singletons (except the settings singleton which is frozen).
- **Fault-tolerant**: WebSocket drop → polling fallback. Broker timeout → circuit breaker + alert. DB locked → retry with backoff.

### 1.4 Execution Modes

| Mode | Description | Real Orders | Real Capital |
|------|-------------|-------------|--------------|
| **LIVE** | Full production execution via Zerodha Kite | YES | YES |
| **PAPER** | Paper broker — simulated fills at mid-price | NO | NO |
| **SHADOW** | Real market data, strategy fires, NO orders placed. Tracks hypothetical P&L drift. | NO | NO |

### 1.5 Design Constraints

- **No naked short options**: Bot only buys calls and puts. Writing options requires exchange-approved margin and is out of scope.
- **NSE only**: Only Nifty index options traded on NSE segment. No stocks, no BankNifty (can be added later via config).
- **Single account**: One Zerodha account per deployment instance.
- **Single process**: No distributed workers. Concurrency is via asyncio, not multiprocessing.
- **IST timezone**: All timestamps stored and displayed in IST (Asia/Kolkata, UTC+5:30).

---

## SECTION 2: TECHNOLOGY STACK

### 2.1 Core Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime (3.11 for faster asyncio, tomllib support) |
| fastapi | 0.111+ | Async HTTP server, health endpoints, WebUI |
| uvicorn | 0.29+ | ASGI server |
| pydantic | 2.7+ | Settings validation, data models |
| pydantic-settings | 2.3+ | Env var loading with aliases |

### 2.2 Broker & Market Data

| Package | Version | Purpose |
|---------|---------|---------|
| kiteconnect | 5.0+ | Zerodha REST + WebSocket SDK |
| httpx | 0.27+ | Async HTTP client for broker REST calls |
| websockets | 12+ | Low-level WebSocket (fallback if kite WS fails) |

### 2.3 Data & Indicators

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2+ | OHLCV DataFrames, rolling windows |
| numpy | 1.26+ | Vectorized indicator math |
| scipy | 1.13+ | Black-Scholes Greeks (norm.cdf, norm.pdf) |

### 2.4 Persistence & Async I/O

| Package | Version | Purpose |
|---------|---------|---------|
| aiosqlite | 0.20+ | Async SQLite for trade journal |
| aiofiles | 24+ | Async file I/O for logs/cache |

### 2.5 Notifications

| Package | Version | Purpose |
|---------|---------|---------|
| python-telegram-bot | 21+ | Telegram bot (async, webhook + polling) |

### 2.6 Observability

| Package | Version | Purpose |
|---------|---------|---------|
| prometheus-client | 0.20+ | Metrics exposition |
| structlog | 24+ | Structured JSON logging |

### 2.7 Infrastructure

| Package | Version | Purpose |
|---------|---------|---------|
| docker | (host) | Container runtime |
| docker-compose | v2 | Multi-service orchestration |
| apscheduler | 3.10+ | Background job scheduler (daily summary, cleanup) |

### 2.8 Testing

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | 8+ | Test runner |
| pytest-asyncio | 0.23+ | Async test support |
| pytest-mock | 3.14+ | Mocking utilities |
| freezegun | 1.4+ | Time mocking for session gate tests |
| respx | 0.21+ | httpx request mocking |

### 2.9 Code Quality

| Package | Version | Purpose |
|---------|---------|---------|
| ruff | 0.4+ | Linting + formatting |
| mypy | 1.10+ | Static type checking |
| pre-commit | 3.7+ | Git hooks |

---

## SECTION 3: COMPLETE DIRECTORY STRUCTURE

```
nifty_scalper_v2/
├── src/
│   ├── main.py                         # FastAPI app factory + uvicorn entrypoint
│   ├── bot.py                          # NiftyScalperBot: top-level orchestrator
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                 # Pydantic BaseSettings — single env loading mechanism
│   │   └── constants.py                # All magic numbers (tick size, lot size, etc.)
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── base.py                     # BrokerGateway ABC
│   │   ├── zerodha.py                  # Kite Connect implementation
│   │   └── paper.py                    # Paper trading (simulated fills at mid)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tick_hub.py                 # Tick fan-out + LRU cache + EventBus
│   │   ├── candle_engine.py            # Real-time OHLCV aggregation per symbol+timeframe
│   │   ├── market_data.py              # MarketDataManager facade (symbol → indicators)
│   │   └── instruments.py              # Symbol/token resolution, strike selection
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── websocket_stream.py         # Kite WebSocket consumer (primary)
│   │   └── polling_stream.py           # REST LTP polling fallback
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── engine.py                   # IndicatorEngine: single source for all indicators
│   │   ├── technical.py                # RSI, VWAP, EMA, SMA, ATR, Bollinger, ADX, CPR
│   │   └── greeks.py                   # Black-Scholes delta, gamma, theta, vega, IV
│   ├── options/
│   │   ├── __init__.py
│   │   ├── universe.py                 # Option chain loader, strike selection, expiry calc
│   │   └── max_pain.py                 # OI-weighted max pain calculation
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                     # Strategy ABC
│   │   ├── signal.py                   # Signal dataclass (canonical)
│   │   ├── manager.py                  # StrategyManager: lifecycle + performance scoring
│   │   ├── runner.py                   # Event-driven strategy execution loop
│   │   └── impl/
│   │       ├── __init__.py
│   │       ├── smc_liquidity.py        # Smart Money Concepts liquidity sweep
│   │       ├── vwap_pro.py             # VWAP crossover + index bias
│   │       ├── rsi_divergence.py       # RSI bullish/bearish divergence
│   │       ├── gamma_scalping.py       # Gamma exposure position sizing
│   │       ├── oi_max_pain.py          # OI max pain strike attraction
│   │       ├── orb_pro.py              # Opening Range Breakout (15-min)
│   │       ├── bb_squeeze.py           # Bollinger Band squeeze + breakout
│   │       ├── cpr_breakout.py         # Central Pivot Range edge breakout
│   │       ├── order_flow.py           # Volume imbalance ratio
│   │       ├── straddle_theta.py       # IV-based theta decay straddle
│   │       ├── trend_momentum.py       # ADX + momentum trend classifier
│   │       └── tuesday_gamma.py        # Tuesday gamma ramp buyer (Thu expiry)
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── order_state.py              # Order state machine (canonical order struct)
│   │   ├── bracket.py                  # SL/TP bracket management (no circular deps)
│   │   ├── lifecycle.py                # Position lifecycle: trail, time exit, partial
│   │   ├── reconciler.py               # Async broker reconciliation loop
│   │   └── router.py                   # LIVE/SHADOW/PAPER execution routing
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── manager.py                  # RiskManager: pre-trade gating
│   │   ├── sizing.py                   # Position sizing formulas (Kelly-fractional)
│   │   ├── limits.py                   # Hard limit definitions (immutable)
│   │   └── session.py                  # Market hours gate (IST-aware)
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── detector.py                 # Market regime classifier (TREND/RANGE/VOLATILE/CALM)
│   │   └── gate.py                     # Regime-based strategy enable/disable
│   ├── journal/
│   │   ├── __init__.py
│   │   ├── db.py                       # aiosqlite pool, migrations, connection management
│   │   ├── trade_log.py                # Trade recording (insert/update)
│   │   └── queries.py                  # Analytics: daily PnL, win rate, drawdown
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── telegram/
│   │   │   ├── __init__.py
│   │   │   ├── bot.py                  # Bot setup, webhook registration, dispatcher
│   │   │   ├── commands/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── status.py           # /status, /health, /uptime
│   │   │   │   ├── trading.py          # /positions, /orders, /pnl, /fills
│   │   │   │   ├── control.py          # /pause, /resume, /mode, /killswitch
│   │   │   │   ├── strategy.py         # /strategies, /regime, /enable, /disable
│   │   │   │   └── admin.py            # /logs, /debug, /metrics, /journal
│   │   │   └── alerts.py               # Proactive alert sender (fills, SL hits, daily summary)
│   │   └── notifier.py                 # Notification facade (decouples bot from Telegram)
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Prometheus Counter/Gauge/Histogram definitions
│   │   ├── health.py                   # /health, /ready endpoints
│   │   └── scheduler.py                # APScheduler: daily summary, cleanup, token refresh
│   └── utils/
│       ├── __init__.py
│       ├── retry.py                    # Exponential backoff decorator
│       ├── rate_limiter.py             # Leaky bucket token counter
│       ├── time_utils.py               # IST helpers: market_open(), is_trading_hours(), etc.
│       └── logging.py                  # structlog setup, JSON renderer
├── tests/
│   ├── conftest.py                     # Shared fixtures: mock broker, fake ticks, test DB
│   ├── unit/
│   │   ├── test_settings.py
│   │   ├── test_tick_hub.py
│   │   ├── test_candle_engine.py
│   │   ├── test_indicator_engine.py
│   │   ├── test_order_state.py
│   │   ├── test_bracket.py
│   │   ├── test_lifecycle.py
│   │   ├── test_risk_manager.py
│   │   ├── test_regime_detector.py
│   │   ├── test_greeks.py
│   │   ├── test_max_pain.py
│   │   └── strategies/
│   │       ├── test_smc_liquidity.py
│   │       ├── test_vwap_pro.py
│   │       ├── test_rsi_divergence.py
│   │       ├── test_gamma_scalping.py
│   │       ├── test_oi_max_pain.py
│   │       ├── test_orb_pro.py
│   │       ├── test_bb_squeeze.py
│   │       ├── test_cpr_breakout.py
│   │       ├── test_order_flow.py
│   │       ├── test_straddle_theta.py
│   │       ├── test_trend_momentum.py
│   │       └── test_tuesday_gamma.py
│   ├── integration/
│   │   ├── test_execution_flow.py      # Signal → order → fill → position
│   │   ├── test_reconciler.py          # Broker sync integration
│   │   └── test_journal.py             # DB write/read round-trip
│   └── e2e/
│       └── test_paper_session.py       # Full paper trading session sim
├── config/
│   └── strategies.yaml                 # Per-strategy overrides (supplements env)
├── ops/
│   ├── monitoring/
│   │   ├── prometheus.yml              # Prometheus scrape config
│   │   └── grafana/
│   │       └── dashboards/
│   │           └── nifty_scalper.json  # Grafana dashboard JSON
│   └── Dockerfile                      # Multi-stage build
├── .env.example                        # All env vars with descriptions
├── requirements.txt                    # Pinned production deps
├── pyproject.toml                      # Build metadata, ruff, mypy config
└── docker-compose.yml                  # bot + prometheus + grafana
```

### 3.1 File Size Constraints

Every source file MUST stay under 1,000 lines. If a file approaches 800 lines, split it. This is enforced by CI via `wc -l src/**/*.py | awk '$1 > 1000 {print; exit 1}'`.

### 3.2 Import Graph Rules

The import dependency graph MUST be a DAG. The following dependency order is enforced (lower layers CANNOT import higher layers):

```
utils → config → broker → data → streaming → indicators → options
      → risk → regime → strategies → execution → journal → notifications → infra → bot → main
```

Circular imports are detected by `import-linter` in CI.

---

## SECTION 4: CONFIGURATION SPEC

### 4.1 Loading Mechanism

**Single mechanism**: `pydantic-settings` `BaseSettings` class in `config/settings.py`. No `os.environ.get()` calls anywhere else in the codebase. Settings are loaded once at startup and the resulting frozen `Settings` instance is passed via dependency injection.

Environment variables are loaded in this priority order (highest to lowest):
1. Actual environment variables (set by Docker/Railway/shell)
2. `.env` file in project root
3. Default values defined in `Settings`

### 4.2 Settings Structure

```python
# config/settings.py — complete settings hierarchy
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,  # Immutable after load
    )

    broker: BrokerSettings
    trading: TradingSettings
    risk: RiskSettings
    lifecycle: LifecycleSettings
    execution: ExecutionSettings
    streaming: StreamingSettings
    strategies: StrategiesSettings
    telegram: TelegramSettings
    infra: InfraSettings
    logging: LoggingSettings
```

### 4.3 Complete Parameter Reference

#### 4.3.1 Broker Settings (`BROKER__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `BROKER__API_KEY` | str | **required** | — | Zerodha Kite API key |
| `BROKER__API_SECRET` | str | **required** | — | Kite API secret (for token generation only) |
| `BROKER__ACCESS_TOKEN` | str | **required** | — | Kite access token (refresh daily) |
| `BROKER__BASE_URL` | str | `https://api.kite.trade` | valid URL | Kite REST base URL |
| `BROKER__WS_URL` | str | `wss://ws.kite.trade` | valid ws:// URL | Kite WebSocket URL |
| `BROKER__TIMEOUT_SECONDS` | float | `10.0` | 1–60 | HTTP request timeout |
| `BROKER__CONNECT_RETRIES` | int | `3` | 1–10 | Retries on connection failure |
| `BROKER__RATE_LIMIT_ORDERS_PER_SEC` | float | `5.0` | 0.1–10 | Zerodha order rate limit |
| `BROKER__RATE_LIMIT_REST_PER_SEC` | float | `10.0` | 0.1–20 | Zerodha REST rate limit |

#### 4.3.2 Trading Settings (`TRADING__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `TRADING__EXECUTION_MODE` | str | `PAPER` | `LIVE`, `SHADOW`, `PAPER` | Execution mode |
| `TRADING__ENABLE_LIVE` | bool | `false` | true/false | Safety override: must be true for LIVE mode |
| `TRADING__INSTRUMENT` | str | `NIFTY` | `NIFTY`, `BANKNIFTY` | Underlying index |
| `TRADING__EXCHANGE` | str | `NFO` | `NFO` | Exchange segment |
| `TRADING__LOT_SIZE` | int | `50` | 1–200 | Options lot size |
| `TRADING__TICK_SIZE` | float | `0.05` | 0.01–1.0 | Minimum price tick |
| `TRADING__MAX_STRIKES_FROM_ATM` | int | `5` | 1–20 | Strike selection radius |
| `TRADING__PREFERRED_EXPIRY` | str | `weekly` | `weekly`, `monthly` | Expiry preference |

#### 4.3.3 Risk Settings (`RISK__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `RISK__MAX_DAILY_TRADES` | int | `30` | 1–100 | Max trades per session |
| `RISK__DAILY_LOSS_PCT` | float | `2.0` | 0.1–10.0 | Max daily loss as % of capital |
| `RISK__MAX_ORDER_NOTIONAL` | float | `200000.0` | 1000–5000000 | Max notional per order (INR) |
| `RISK__MAX_CONCURRENT_POSITIONS` | int | `3` | 1–10 | Max open positions simultaneously |
| `RISK__MAX_POSITIONS_PER_SYMBOL` | int | `1` | 1–5 | Max positions per option symbol |
| `RISK__CONSECUTIVE_LOSS_COOLDOWN` | int | `3` | 0–10 | Pause after N consecutive losses |
| `RISK__COOLDOWN_MINUTES` | int | `30` | 5–120 | Duration of consecutive loss cooldown |
| `RISK__MIN_OPTION_OI` | int | `100` | 0–10000 | Reject options with OI below this |
| `RISK__MAX_SPREAD_PCT` | float | `3.0` | 0.1–20.0 | Reject options if spread > X% of mid |
| `RISK__MIN_OPTION_PRICE` | float | `5.0` | 1.0–100.0 | Reject options below this price |
| `RISK__MAX_OPTION_PRICE` | float | `500.0` | 10.0–5000.0 | Reject options above this price |
| `RISK__VIX_HIGH_THRESHOLD` | float | `20.0` | 10–50 | VIX above this → reduce sizing 50% |
| `RISK__VIX_EXTREME_THRESHOLD` | float | `30.0` | 15–80 | VIX above this → stop trading |
| `RISK__CAPITAL` | float | `1000000.0` | 10000–∞ | Total capital (INR) for sizing |

#### 4.3.4 Lifecycle Settings (`LIFECYCLE__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `LIFECYCLE__TP1_R` | float | `1.0` | 0.5–5.0 | TP1 in multiples of initial risk (R) |
| `LIFECYCLE__TP1_PARTIAL` | float | `0.6` | 0.1–0.9 | Fraction of position to close at TP1 |
| `LIFECYCLE__TP2_R_TREND` | float | `1.8` | 1.0–10.0 | TP2 R-multiple in TREND regime |
| `LIFECYCLE__TP2_R_RANGE` | float | `1.4` | 1.0–5.0 | TP2 R-multiple in RANGE regime |
| `LIFECYCLE__TRAIL_ATR_MULT` | float | `0.8` | 0.3–3.0 | Trailing stop = ATR × this multiplier |
| `LIFECYCLE__TIME_STOP_MIN` | int | `12` | 5–60 | Force exit if position open > N minutes |
| `LIFECYCLE__TRAIL_ACTIVATION_R` | float | `1.0` | 0.5–3.0 | Start trailing after price moves N×R |

#### 4.3.5 Execution Settings (`EXECUTION__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `EXECUTION__ORDER_TYPE` | str | `MARKET` | `MARKET`, `LIMIT` | Default order type for entries |
| `EXECUTION__LIMIT_OFFSET_TICKS` | int | `2` | 0–20 | Limit order offset in ticks from mid |
| `EXECUTION__SLIPPAGE_MODEL` | str | `SPREAD_HALF` | `ZERO`, `SPREAD_HALF`, `FIXED` | Paper mode slippage model |
| `EXECUTION__RECONCILE_INTERVAL_SEC` | int | `30` | 5–300 | Broker reconciliation frequency |
| `EXECUTION__MAX_FILL_WAIT_SEC` | int | `60` | 10–300 | Cancel unfilled limit orders after N sec |

#### 4.3.6 Streaming Settings (`STREAMING__*`)

| Env Variable | Type | Default | Range/Options | Description |
|---|---|---|---|---|
| `STREAMING__WEBSOCKET_DISABLED` | bool | `false` | true/false | Force polling mode (Railway compat) |
| `STREAMING__POLLING_INTERVAL_MS` | int | `1000` | 200–5000 | REST poll interval in milliseconds |
| `STREAMING__WARMUP_TICKS` | int | `100` | 10–500 | Ticks before strategies activate |
| `STREAMING__WARMUP_CANDLES` | int | `20` | 5–100 | Min candles before indicator valid |
| `STREAMING__MAX_RECONNECT_ATTEMPTS` | int | `10` | 1–100 | WS reconnect attempts before alert |
| `STREAMING__RECONNECT_BACKOFF_BASE` | float | `2.0` | 1.1–10.0 | Exponential backoff base |

#### 4.3.7 Per-Strategy Settings (`{STRATEGY_NAME}__*`)

Each strategy has these overridable parameters (replace `{NAME}` with strategy name in UPPER_SNAKE_CASE):

| Env Variable | Type | Default | Description |
|---|---|---|---|
| `{NAME}__ENABLED` | bool | `true` | Enable/disable strategy |
| `{NAME}__MIN_CONFIDENCE` | float | `0.6` | Minimum signal confidence (0–1) |
| `{NAME}__COOLDOWN_SEC` | int | `300` | Seconds between signals from this strategy |
| `{NAME}__MAX_DAILY_SIGNALS` | int | `5` | Max signals per day from this strategy |

Strategy name keys: `SMC_LIQUIDITY`, `VWAP_PRO`, `RSI_DIVERGENCE`, `GAMMA_SCALPING`, `OI_MAX_PAIN`, `ORB_PRO`, `BB_SQUEEZE`, `CPR_BREAKOUT`, `ORDER_FLOW`, `STRADDLE_THETA`, `TREND_MOMENTUM`, `TUESDAY_GAMMA`

#### 4.3.8 Telegram Settings (`TELEGRAM__*`)

| Env Variable | Type | Default | Description |
|---|---|---|---|
| `TELEGRAM__BOT_TOKEN` | str | None | Bot token from @BotFather |
| `TELEGRAM__CHAT_ID` | int | None | Authorized operator chat ID |
| `TELEGRAM__WEBHOOK_URL` | str | None | Public HTTPS URL for webhook mode |
| `TELEGRAM__USE_POLLING` | bool | `true` | Use polling (true) or webhook (false) |
| `TELEGRAM__RATE_LIMIT_PER_MIN` | int | `20` | Max Telegram API calls per minute |

#### 4.3.9 Infra Settings (`INFRA__*`)

| Env Variable | Type | Default | Description |
|---|---|---|---|
| `INFRA__METRICS_PORT` | int | `9090` | Prometheus metrics exposition port |
| `INFRA__API_PORT` | int | `8000` | FastAPI server port |
| `INFRA__DB_PATH` | str | `./data/journal.db` | SQLite database file path |
| `INFRA__LOG_LEVEL` | str | `INFO` | Log level: DEBUG/INFO/WARNING/ERROR |
| `INFRA__LOG_FORMAT` | str | `json` | Log format: `json` or `console` |
| `INFRA__DAILY_SUMMARY_TIME` | str | `15:20` | IST time for daily summary Telegram message |

### 4.4 Security Notes

- `BROKER__ACCESS_TOKEN` expires daily at midnight. Token refresh is out of scope for this bot (use a separate token refresh script or Kite's OAuth flow on a schedule).
- Never log `BROKER__ACCESS_TOKEN` or `BROKER__API_SECRET`. The settings module should mask these in `__repr__`.
- The Telegram `CHAT_ID` acts as the authorization mechanism. The bot silently ignores all messages from other chat IDs.

---

## SECTION 5: DATA MODELS

All data models use Python `dataclasses` or `pydantic.BaseModel` depending on use case:
- **Pydantic**: Models that cross network boundaries (API responses, Telegram payloads, config)
- **Dataclasses** with `__slots__`: High-frequency internal models (Tick, OHLCV, Signal) for performance

### 5.1 Tick

```python
# data/tick_hub.py
@dataclass(slots=True, frozen=True)
class Tick:
    symbol: str           # NSE trading symbol, e.g. "NIFTY2461023500CE"
    instrument_token: int # Zerodha numeric token
    ltp: float            # Last traded price
    bid: float            # Best bid (0.0 if unavailable)
    ask: float            # Best ask (0.0 if unavailable)
    volume: int           # Cumulative day volume
    oi: int               # Open interest (options only)
    timestamp: datetime   # UTC timestamp of tick
    is_backfill: bool = False  # True if replayed (not live) — CRITICAL for warmup logic

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ltp

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.mid > 0:
            return (self.spread / self.mid) * 100.0
        return 0.0
```

### 5.2 OHLCV

```python
# data/candle_engine.py
@dataclass(slots=True)
class OHLCV:
    symbol: str
    timeframe: str        # "1m", "5m", "15m", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int               # OI at candle close (options)
    timestamp: datetime   # Candle open time (UTC)
    is_complete: bool     # False while candle still building
```

### 5.3 Indicators

```python
# indicators/engine.py
@dataclass(slots=True)
class IndicatorSnapshot:
    symbol: str
    timestamp: datetime
    # Price
    ltp: float
    vwap: float
    # Trend
    ema_9: float | None
    ema_21: float | None
    ema_50: float | None
    adx: float | None
    adx_plus_di: float | None
    adx_minus_di: float | None
    # Momentum
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    # Volatility
    atr_14: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_mid: float | None
    bb_width: float | None        # (upper - lower) / mid
    bb_squeeze: bool | None       # bb_width < 20th percentile
    # Volume
    volume_sma_20: float | None
    volume_ratio: float | None    # current_volume / volume_sma_20
    # Options-specific
    iv: float | None              # Implied volatility (annualized)
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    # Session
    day_high: float | None
    day_low: float | None
    opening_range_high: float | None   # 15-min ORB
    opening_range_low: float | None
    cpr_top: float | None              # Central Pivot Range
    cpr_pivot: float | None
    cpr_bottom: float | None
    # Market
    vix: float | None             # India VIX
    max_pain_strike: float | None
    is_valid: bool = True         # False if insufficient data
```

### 5.4 Signal

```python
# strategies/signal.py
from enum import Enum

class Direction(str, Enum):
    BUY_CALL = "BUY_CALL"    # Buy CE (bullish)
    BUY_PUT  = "BUY_PUT"     # Buy PE (bearish)

class SignalStrength(str, Enum):
    WEAK     = "WEAK"         # confidence 0.5–0.65
    MODERATE = "MODERATE"     # confidence 0.65–0.80
    STRONG   = "STRONG"       # confidence > 0.80

@dataclass(slots=True)
class Signal:
    strategy_name: str
    symbol: str               # Underlying symbol (e.g. "NIFTY")
    direction: Direction
    confidence: float         # 0.0–1.0
    entry_price: float        # Estimated option premium at entry
    sl_price: float           # Stop loss premium level
    tp1_price: float          # Take profit 1 premium level
    tp2_price: float          # Take profit 2 premium level
    quantity: int             # Number of lots
    strike: float             # Recommended strike
    expiry: date              # Recommended expiry
    option_type: str          # "CE" or "PE"
    regime: str               # Regime at signal time
    timestamp: datetime       # Signal generation time (UTC)
    metadata: dict            # Strategy-specific extras (raw indicator values)

    @property
    def strength(self) -> SignalStrength:
        if self.confidence >= 0.80:
            return SignalStrength.STRONG
        elif self.confidence >= 0.65:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def risk_per_lot(self) -> float:
        return abs(self.entry_price - self.sl_price) * LOT_SIZE

    @property
    def r_multiple_tp1(self) -> float:
        if self.risk_per_lot > 0:
            return abs(self.tp1_price - self.entry_price) / abs(self.entry_price - self.sl_price)
        return 0.0
```

### 5.5 Order

```python
# execution/order_state.py
class OrderStatus(str, Enum):
    PENDING         = "PENDING"
    SUBMITTED       = "SUBMITTED"
    OPEN            = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED          = "FILLED"
    CANCELLED       = "CANCELLED"
    REJECTED        = "REJECTED"

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"
    SL_M   = "SL-M"

@dataclass
class Order:
    order_id: str               # Internal UUID
    broker_order_id: str | None # Broker-assigned ID (None until submitted)
    symbol: str                 # Full option symbol
    side: OrderSide
    quantity: int               # Total lots
    filled_quantity: int        # Filled so far
    price: float                # Limit price (0.0 for MARKET)
    trigger_price: float        # SL trigger price (0.0 if not SL)
    order_type: OrderType
    status: OrderStatus
    strategy_name: str          # Which strategy generated this
    position_id: str            # Links to position
    average_price: float        # Average fill price
    created_at: datetime
    submitted_at: datetime | None
    filled_at: datetime | None
    cancelled_at: datetime | None
    reject_reason: str | None
    fills: list[Fill]
```

### 5.6 Fill

```python
@dataclass(slots=True, frozen=True)
class Fill:
    fill_id: str
    order_id: str
    quantity: int
    price: float
    timestamp: datetime
    is_partial: bool
```

### 5.7 Position

```python
# execution/order_state.py
class PositionSide(str, Enum):
    LONG = "LONG"    # Bought options (calls or puts)
    SHORT = "SHORT"  # Written options — NEVER used in v2

@dataclass
class Position:
    position_id: str
    symbol: str               # Full option symbol
    underlying: str           # "NIFTY"
    option_type: str          # "CE" or "PE"
    strike: float
    expiry: date
    side: PositionSide        # Always LONG in v2
    quantity: int             # Current open lots
    initial_quantity: int     # Original lots (before partials)
    entry_price: float        # Average entry premium
    current_price: float      # Latest LTP of option
    pnl_unrealized: float
    pnl_realized: float       # From closed partial lots
    strategy_name: str
    regime_at_entry: str
    opened_at: datetime
    last_updated_at: datetime
    is_closed: bool = False
```

### 5.8 Bracket

```python
# execution/bracket.py
@dataclass
class Bracket:
    bracket_id: str
    position_id: str
    symbol: str
    side: str                 # "BUY" or "SELL" — normalized, NEVER "LONG"/"SHORT"
    initial_quantity: int
    remaining_quantity: int
    entry_price: float
    sl_price: float           # Current stop-loss level
    initial_sl_price: float   # Original SL (for R-multiple calculation)
    tp1_price: float
    tp2_price: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    trail_active: bool = False
    trail_price: float | None = None    # Current trailing stop level
    peak_price: float | None = None     # Highest favorable price seen
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_evaluated_at: datetime | None = None

    def initial_risk(self) -> float:
        return abs(self.entry_price - self.initial_sl_price)

    def current_r_multiple(self, current_price: float) -> float:
        risk = self.initial_risk()
        if risk <= 0:
            return 0.0
        if self.side == "BUY":
            return (current_price - self.entry_price) / risk
        return (self.entry_price - current_price) / risk
```

### 5.9 Trade (Journal Record)

```python
# journal/trade_log.py
@dataclass
class Trade:
    trade_id: str
    strategy_name: str
    symbol: str
    underlying: str
    option_type: str          # "CE" or "PE"
    strike: float
    expiry: date
    direction: str            # "BUY_CALL" or "BUY_PUT"
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    brokerage: float
    net_pnl: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    exit_reason: str          # "TP1", "TP2", "SL", "TRAIL", "TIME", "MANUAL", "EOD"
    regime_at_entry: str
    confidence_at_entry: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: float
    r_multiple_achieved: float  # (exit - entry) / (entry - sl) for longs
    is_winner: bool
```

### 5.10 MarketRegime

```python
# regime/detector.py
class RegimeType(str, Enum):
    TREND    = "TREND"     # ADX > 25, clear directional move
    RANGE    = "RANGE"     # ADX < 20, price oscillating
    VOLATILE = "VOLATILE"  # VIX > threshold OR ATR spike
    CALM     = "CALM"      # Low VIX + low ATR + no ADX trend

@dataclass(slots=True)
class MarketRegime:
    regime: RegimeType
    confidence: float     # 0.0–1.0 how confident the classifier is
    adx: float
    vix: float
    atr_ratio: float      # Current ATR / 20-day average ATR
    direction: str | None # "BULLISH" or "BEARISH" (only set in TREND regime)
    detected_at: datetime
```

### 5.11 RiskState

```python
# risk/manager.py
@dataclass
class RiskState:
    daily_pnl: float              # Running P&L for today (INR)
    daily_pnl_pct: float          # As % of capital
    trade_count: int              # Trades executed today
    consecutive_losses: int       # Current streak of losses
    is_paused: bool               # Manually paused via Telegram
    is_daily_limit_hit: bool      # Daily loss limit breached
    is_cooldown_active: bool      # Consecutive loss cooldown active
    cooldown_until: datetime | None
    open_positions_count: int
    total_exposure: float         # Sum of all position notionals (INR)
    last_updated: datetime
```

---

## SECTION 6: CORE ARCHITECTURE AND FLOW

### 6.1 Startup Sequence

The bot starts in `bot.py :: NiftyScalperBot.startup()`. Steps execute sequentially; any fatal failure emits an alert and exits.

```
1.  Load Settings (config/settings.py)
       └── Validate all env vars via Pydantic
       └── Fail fast if BROKER__API_KEY missing or EXECUTION_MODE invalid

2.  Init Structured Logging (utils/logging.py)
       └── structlog with JSON renderer in prod, console renderer in dev

3.  Init Database (journal/db.py)
       └── Create SQLite file if not exists
       └── Run migrations (idempotent CREATE TABLE IF NOT EXISTS)
       └── Verify write access

4.  Init Metrics (infra/metrics.py)
       └── Register all Prometheus counters/gauges/histograms
       └── Start metrics HTTP server on INFRA__METRICS_PORT

5.  Connect Broker (broker/zerodha.py)
       └── Verify access token via GET /user/profile (timeout: 10s)
       └── If PAPER mode → instantiate PaperBroker instead
       └── Fetch instruments list (NSE_FO)

6.  Sync Instrument Universe (data/instruments.py)
       └── Load full NFO instrument CSV
       └── Build symbol → token mapping for Nifty options
       └── Identify current weekly + monthly expiry tokens

7.  Init DataHub (data/tick_hub.py)
       └── Create EventBus
       └── Initialize tick cache (symbol → latest Tick)
       └── Set warmup_complete = False

8.  Start Streaming (streaming/websocket_stream.py OR polling_stream.py)
       └── Subscribe to: Nifty spot, India VIX, ATM ±5 strikes CE+PE
       └── Begin feeding ticks into DataHub
       └── Warmup: count ticks per symbol until STREAMING__WARMUP_TICKS reached
       └── CRITICAL: is_backfill=True on all ticks until warmup_complete=True

9.  Build Indicator Engine (indicators/engine.py)
       └── Subscribe to candle_complete events from CandleEngine
       └── Recalculate indicators on each completed candle

10. Init CandleEngine (data/candle_engine.py)
       └── Subscribe to tick events
       └── Build 1m, 5m, 15m OHLCV bars per symbol

11. Init Strategy Instances (strategies/manager.py)
       └── Instantiate all 12 strategy classes
       └── Pass shared indicator_engine reference
       └── Apply per-strategy settings from config

12. Init Risk Manager (risk/manager.py)
       └── Load today's trades from journal
       └── Calculate starting daily_pnl, trade_count

13. Init Order/Bracket/Lifecycle Managers (execution/)
       └── Load open positions from broker reconciliation
       └── Attach brackets to recovered positions

14. Start Reconciler (execution/reconciler.py)
       └── Background task: poll broker every RECONCILE_INTERVAL_SEC

15. Start Strategy Runner (strategies/runner.py)
       └── 100ms event loop begins
       └── Waits for warmup_complete before generating signals

16. Start FastAPI Server (main.py)
       └── Mounts health, metrics, strategy status endpoints

17. Start Telegram Bot (notifications/telegram/bot.py)
       └── Register all command handlers
       └── Start webhook or polling

18. Emit "READY" — send Telegram message: "NiftyScalperBot v2 READY | Mode: {mode}"
```

### 6.2 Main Event Loop (100ms cadence)

```python
async def _run_loop(self) -> None:
    while self._running:
        loop_start = time.monotonic()
        try:
            # 1. Session gate — skip outside trading hours
            if not self.session_gate.is_trading_now():
                await asyncio.sleep(1.0)
                continue

            # 2. Skip if warmup not complete
            if not self.tick_hub.warmup_complete:
                await asyncio.sleep(0.1)
                continue

            # 3. Get latest snapshots for tracked symbols
            snapshots = self.tick_hub.get_all_latest()

            # 4. Update candle engine (ticks fed via EventBus, candles built asynchronously)
            # Candle engine updates happen in the tick handler, not here

            # 5. Check regime (cached, recalculated every 5 minutes)
            regime = await self.regime_detector.get_current()

            # 6. Generate signals from all enabled strategies
            signals: list[Signal] = []
            for strategy in self.strategy_manager.active_strategies():
                for symbol, indicators in self.indicator_engine.get_all_snapshots().items():
                    if indicators.is_valid:
                        sig = strategy.generate_signal(symbol, indicators, regime)
                        if sig:
                            signals.append(sig)

            # 7. Orchestrator filtering
            filtered = self._filter_signals(signals)

            # 8. Preflight validation
            validated = [s for s in filtered if await self._preflight(s)]

            # 9. Route to execution
            for signal in validated:
                [DEPRECATED] do not add execution_router.route(signal) in current runtime architecture

            # 10. Update lifecycle (SL/TP/trail evaluation)
            await self.lifecycle_manager.evaluate_all()

            # 11. Update metrics
            self._record_loop_metrics(len(signals), len(validated))

        except Exception as exc:
            self.logger.error("loop_error", error=str(exc), exc_info=True)
            self.metrics.loop_errors.inc()

        # Maintain 100ms cadence
        elapsed = time.monotonic() - loop_start
        sleep_time = max(0.0, 0.1 - elapsed)
        await asyncio.sleep(sleep_time)
```

### 6.3 Signal Filtering (Orchestrator Logic)

The orchestrator applies these filters in order. A signal is dropped if ANY filter rejects it:

```
1. Direction lock: If already have a LONG position → reject all BUY_CALL signals for same underlying
2. Capital headroom: total_exposure + new_notional < RISK__MAX_ORDER_NOTIONAL
3. Concurrent limit: open_positions < RISK__MAX_CONCURRENT_POSITIONS
4. Strategy cooldown: strategy.last_signal_at + cooldown_sec < now
5. Daily signal limit: strategy.daily_signal_count < MAX_DAILY_SIGNALS
6. Regime gate: regime.allows(strategy.name) == True
7. Risk manager gate: risk_manager.can_trade() == True
8. Duplicate suppression: no identical (symbol + direction) signal in last 60 seconds
```

### 6.4 Order State Machine

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
  [Signal arrives]  │                                         ▼
        │           │         ┌──────────────────┐      CANCELLED
        ▼           │         │                  │
    PENDING ──submit──▶ SUBMITTED ──ack──▶ OPEN ──fill──▶ PARTIALLY_FILLED
                                │              │                │
                                │              │       more fills│
                                │              │                ▼
                                │              └────────▶ FILLED
                                │
                                └──reject──▶ REJECTED
```

Valid transitions (all others raise `InvalidTransitionError`):
- `PENDING → SUBMITTED` (on broker submit)
- `SUBMITTED → OPEN` (broker acknowledged)
- `SUBMITTED → REJECTED` (broker rejected)
- `OPEN → PARTIALLY_FILLED` (partial fill)
- `OPEN → FILLED` (full fill in one go)
- `OPEN → CANCELLED` (cancel request confirmed)
- `PARTIALLY_FILLED → PARTIALLY_FILLED` (more partials)
- `PARTIALLY_FILLED → FILLED` (remainder filled)
- `PARTIALLY_FILLED → CANCELLED` (cancelled after partials — rare)

### 6.5 Position Lifecycle State Machine

```
OPEN ──────────────────────────────────────────────────────────────────┐
  │                                                                     │
  ├─ Tick arrives → evaluate_all()                                      │
  │    │                                                                │
  │    ├─ price ≤ sl_price?              → EXIT_SL (market order)       │
  │    │                                                                │
  │    ├─ price ≥ tp1_price AND !tp1_hit → PARTIAL EXIT (60% qty)      │
  │    │                                    tp1_hit = True              │
  │    │                                    activate trailing stop      │
  │    │                                                                │
  │    ├─ price ≥ tp2_price AND tp1_hit  → EXIT_TP2 (close remaining)  │
  │    │                                                                │
  │    ├─ trail_active AND price ≤ trail_price → EXIT_TRAIL (market)   │
  │    │                                                                │
  │    ├─ now - opened_at > TIME_STOP_MIN → EXIT_TIME (market)         │
  │    │                                                                │
  │    └─ /exit command from Telegram   → EXIT_MANUAL (market)         │
  │                                                                     │
  └─────────────────────────────────────────────── CLOSED ◄────────────┘
                                                      │
                                               record Trade in journal
                                               update RiskState
                                               emit Telegram alert
```

**Trailing Stop Logic:**
```
When tp1_hit == True:
    peak_price = max(peak_price, current_price)
    trail_price = peak_price - (atr_14 × LIFECYCLE__TRAIL_ATR_MULT)
    trail_price = max(trail_price, entry_price)  # never trail below breakeven
    if current_price ≤ trail_price:
        → EXIT_TRAIL
```

### 6.6 WebSocket Tick Flow

```
KiteWebSocket.on_tick()
      │
      ▼
  TickHub.on_raw_tick(raw_dict)
      │
      ├── Parse to Tick dataclass
      ├── Validate (ltp > 0, timestamp reasonable)
      ├── Update tick_cache[symbol] = tick
      ├── Feed to CandleEngine.on_tick(tick)
      │       └── Update current bar OHLCV
      │           └── If bar complete → emit candle_complete event
      │                   └── IndicatorEngine.on_candle(candle)
      │                           └── Recalculate indicators
      │                           └── Store in indicator_cache[symbol]
      │
      └── Emit tick_event to all EventBus subscribers
              └── strategies/runner.py collects for next loop iteration
```

### 6.7 Startup Warmup Guard (Critical Fix)

The existing bot had a critical bug where warmup state didn't distinguish live ticks from backfill. The fix:

```python
# streaming/websocket_stream.py
class WebSocketStream:
    def __init__(self, tick_hub: TickHub, warmup_ticks: int):
        self._warmup_ticks = warmup_ticks
        self._tick_counts: dict[str, int] = {}

    def _on_tick(self, ws, ticks: list[dict]) -> None:
        for raw_tick in ticks:
            symbol = self._token_to_symbol(raw_tick["instrument_token"])
            count = self._tick_counts.get(symbol, 0) + 1
            self._tick_counts[symbol] = count
            # Mark as backfill until we have enough live ticks
            is_live = count > self._warmup_ticks
            tick = Tick(..., is_backfill=not is_live)
            self.tick_hub.on_tick(tick)

        # Check if all subscribed symbols have warmed up
        if not self.tick_hub.warmup_complete:
            if all(c > self._warmup_ticks for c in self._tick_counts.values()):
                self.tick_hub.set_warmup_complete()
                logger.info("warmup_complete", tick_counts=self._tick_counts)
```

Strategies check `indicators.is_valid` (set False during warmup) before generating signals.

### 6.8 Execution Router

```python
# execution/router.py
class ExecutionRouter:  # DEPRECATED - not part of current runtime architecture
    async def route(self, signal: Signal) -> None:
        if self.mode == ExecutionMode.LIVE:
            await self._live_execute(signal)
        elif self.mode == ExecutionMode.PAPER:
            await self._paper_execute(signal)
        elif self.mode == ExecutionMode.SHADOW:
            await self._shadow_track(signal)  # No order, just log + notify

    async def _live_execute(self, signal: Signal) -> None:
        # Resolve exact option symbol + token
        option_symbol = await self.instruments.resolve_option(
            signal.underlying, signal.strike, signal.expiry, signal.option_type
        )
        # Place entry order
        order = await self.broker.place_order(option_symbol, "BUY", signal.quantity, OrderType.MARKET)
        # Create bracket
        bracket = self.bracket_manager.create_bracket(order, signal)
        # Start lifecycle monitoring
        self.lifecycle_manager.register(bracket)
        # Notify
        await self.notifier.send_fill_alert(order, signal)
```

---
## SECTION 7: PER-FILE IMPLEMENTATION SPECS