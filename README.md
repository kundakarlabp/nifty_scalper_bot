--- README.md
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

## Runtime Contract/Data Architecture

| Layer | Owner | Responsibility |
|---|---|---|
| Contract / instrument SSOT | `src/nifty_scalper_bot/core/instrument_manager.py` | Loads broker instruments, owns NIFTY spot token, active future, nearest option expiry, ATM CE/PE, nearby option universe, and symbol-token mappings. |
| Runtime basket commit | `src/nifty_scalper_bot/core/app.py` | Commits `ActiveContractBasket` and propagates it to MDM, DataHub, runner, and StrategyManager. |
| Market data / hydration | `src/nifty_scalper_bot/data/market_data_manager.py` | Subscribes `basket.all_tokens`, preserves quote/depth/OI metadata, and reports hydration readiness. |
| Bars | `src/nifty_scalper_bot/data/candle_engine.py` | Builds and validates tick-to-OHLC bars. |
| Strategy read facade | `src/nifty_scalper_bot/data/data_hub.py` | Exposes read-only quote, OHLC, OI/IV/greeks, and active basket context. |
| Strategy evaluation | `src/nifty_scalper_bot/core/strategy_manager.py`, `src/nifty_scalper_bot/strategies/*` | Consumes prepared context and evaluates NIFTY option symbols only. |
| Execution | `src/nifty_scalper_bot/execution/*` | Trades NIFTY options only after risk and execution safety gates pass. |

Runtime flow:

```text
InstrumentManager.load()
→ InstrumentManager.get_active_nifty_contracts()
→ app commits ActiveContractBasket
→ MDM subscribes basket.all_tokens
→ MDM hydrates quote/OHLC/depth/OI
→ CandleEngine builds bars
→ DataHub exposes reads
→ StrategyManager evaluates option symbols only
→ Execution trades options only
```

## Runtime Troubleshooting

- `active_future_unresolved`: futures context is unavailable; option trading can continue when spot and option context are ready.
- `option_token_missing`: the selected option is absent from the broker instrument dump or token map.
- `basket_hydration_failed`: MDM could not seed required quote/OHLC state for selected CE/PE.
- `option_ohlc_insufficient`: selected option bars are not ready for strategy evaluation.
- `candidate_not_selected_or_near_atm`: strategy candidate is outside the committed active basket.
- `direction_context_missing_live`: spot context is missing/stale for live directional evaluation.

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
| `STREAM__MODE` | Market data streaming mode (`websocket` or `poll`) | `websocket` |
| `WEBSOCKET__DISABLED` | Disable websocket transport when `true` | `false` |
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
| `SESSION_ALLOW_OUT_OF_HOURS` | Allow strategy startup and checks outside market hours for testing | `false` |
| `NIFTY_FALLBACK_LTP` | Fallback NIFTY spot used when live ticks are unavailable off-hours | `24000` |
| `MARKETDATA_PRIMARY_SOURCE` | Primary market-data source | `websocket` |
| `MARKETDATA_REST_FALLBACK_ENABLED` | Allow REST quote fallback | `true` |
| `MARKETDATA_REST_FALLBACK_MODE` | REST fallback activation policy | `after_ws_degraded` |
| `MARKETDATA_SUPPRESS_REST_BEFORE_WS` | Suppress REST quote before WS start in LIVE | `true` |
| `MARKETDATA_REQUIRE_WS_SPOT_FOR_LIVE` | Require WS spot proof for LIVE startup | `true` |
| `MARKETDATA_ALLOW_SYNTHETIC_SPOT_IN_LIVE` | Allow synthetic spot in LIVE | `false` |
| `NIFTY_INTERNAL_SYMBOL` | Internal bot NIFTY symbol | `NSE:NIFTY` |
| `NIFTY_ZERODHA_QUOTE_SYMBOL` | Zerodha REST quote symbol for NIFTY index | `NSE:NIFTY 50` |
| `NIFTY_SPOT_TOKEN` | NIFTY spot token | `256265` |
| `NIFTY_EXCHANGE` | Spot exchange code | `NSE` |
| `STARTUP_WAIT_FOR_WS_SPOT_SECONDS` | Max wait for fresh startup WS spot tick | `15` |
| `STARTUP_SPOT_MAX_AGE_SECONDS` | Max accepted age for startup spot tick | `120` |
| `STARTUP_BLOCK_LIVE_IF_NO_WS_SPOT` | Block LIVE arming when no fresh WS spot | `true` |
| `QUOTE_CANONICALIZE_INDEX_SYMBOLS` | Canonicalize index symbols for quote calls | `true` |
| `QUOTE_MARK_403_AS_REST_DEGRADED_ONLY` | Treat quote 403 as quote-path degradation only | `true` |
| `QUOTE_DO_NOT_BLOCK_IF_WS_HEALTHY` | Allow readiness with WS proof when REST quote degraded | `true` |
| `LOG_THROTTLE_QUOTE_ERRORS_SECONDS` | Quote error log throttle interval | `120` |
| `LOG_MARKETDATA_DECISIONS` | Emit market-data policy decision logs | `true` |

> Note:
> - Internal bot symbol remains `NSE:NIFTY`.
> - Zerodha REST quote symbol for the NIFTY index is `NSE:NIFTY 50`.
> - REST quote fallback is intentionally suppressed before WebSocket startup in LIVE mode.

### Live trading toggles

For integration tests that exercise the live order routing code paths, export the following environment variables to avoid `live trading disabled` guard rails:

```bash
export ENABLE_LIVE=1
export SESSION_ALLOW_OUT_OF_HOURS=true  # optional: bypass market hours check during testing
export NIFTY_FALLBACK_LTP=24000        # optional: fallback spot for off-hours symbol selection
```

## License

MIT License.

+++ README.md (修改后)
# Nifty Scalper Bot

A production-grade, modular intraday trading system for Nifty index options with regime-adaptive strategies, multi-layer risk management, and real-time observability.

## Architecture Overview

The bot is built on a **layered architecture** with clear separation of concerns:

```
+------------------------------------------------------------------+
|                    FastAPI HTTP Layer                            |
|  (Health checks, Telegram webhooks, Admin endpoints)             |
+------------------------------------------------------------------+
                              |
+------------------------------------------------------------------+
|                   NiftyScalperApp (Core Orchestrator)            |
|  - Lifecycle management (start/stop/reconcile)                   |
|  - Component wiring & dependency injection                       |
|  - Background task supervision                                   |
+------------------------------------------------------------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
+-------v--------+   +-------v--------+   +-------v--------+
|  Data Layer    |   | Strategy Layer |   | Execution Layer|
|  - WebSocket   |   | - Elite Strat  |   | - OrderManager |
|  - Polling     |   | - ORB          |   | - BracketManager|
|  - Instruments |   | - Regime Adapt |   | - Entry/Exit   |
|  - Cache       |   | - Signal Gen   |   | - Reconcile    |
+----------------+   +----------------+   +----------------+
        |                     |                     |
        +---------------------+---------------------+
                              |
+------------------------------------------------------------------+
|                    Broker Abstraction Layer                      |
|  - Zerodha Kite REST Client                                      |
|  - Instrument Resolution                                         |
|  - Margin & Position Sync                                        |
+------------------------------------------------------------------+
```

## Core Features

### Market Data Infrastructure
- **Dual-mode streaming**: WebSocket (Zerodha Kite Connect) with automatic fallback to REST polling
- **Resilient streamer**: Watchdog-based health monitoring, exponential backoff reconnection
- **Quote cache**: TTL-based LRU cache with staleness guards (QUOTE_STALE_THRESHOLD_MS)
- **Instrument resolver**: On-demand symbol-to-token resolution with SQLite-backed caching
- **Data freshness validation**: Automated assessment of tick latency and gap detection

### Strategy Engine
- **Multi-strategy orchestrator**: Concurrent execution of multiple alpha models
- **Elite Strategies**: Production-ready implementations including:
  - Opening Range Breakout (ORB) with volume confirmation
  - Regime-Adaptive Mean Reversion (VWAP-based)
  - Tuesday Gamma Buyer (expiry-day theta decay capture)
  - Premium Decay Scanner (IV percentile filtering)
- **Signal arbitration**: Confidence-weighted signal fusion with conflict resolution
- **Market regime detection**: Hidden Markov Model + heuristic classification (TREND/RANGE/TRANSITION)
- **Indicator engine**: RSI, ATR, VWAP, Bollinger Bands, SuperTrend with bar-building

### Risk Management Stack
- **Pre-trade validation**: Multi-gate preflight checks before order submission
  - Capital adequacy verification
  - Daily trade count limits
  - Notional exposure caps
  - Short-selling permissions
  - Concentration limits per symbol
- **Position sizing**: Dynamic sizing based on:
  - Volatility-adjusted position sizing (ATR-based)
  - Time-of-day scaling (reduced size in afternoon session)
  - Regime-aware allocation (larger size in TREND, smaller in RANGE)
  - Per-trade capital cap percentage (RISK_PER_TRADE_CAP_PCT)
- **Circuit breakers**:
  - Max drawdown halt (daily loss threshold)
  - Consecutive loss limiter
  - Rate-limit exhaustion protection
- **Session gates**: Market hours enforcement with optional test overrides

### Execution System
- **Order queue**: Priority-based FIFO with age tracking and source attribution
- **Lifecycle manager**: Bracket-order orchestration with:
  - TP1 partial exit (configurable fraction at 1R)
  - TP2 full exit (regime-dependent: 1.8R trend / 1.4R range)
  - ATR-based trailing stop activation post-TP1
  - Time-based stop (auto-exit after LIFECYCLE_TIME_STOP_MIN)
  - Gamma-scaling on expiry Tuesdays
- **Shadow paper trader**: Real-time PnL simulation against live fills for drift detection
- **Post-fill monitor**: Reconciliation loop comparing internal state vs broker snapshot
- **Safe order manager**: Idempotent order submission with retry logic and reject handling
- **Execution modes**:
  - LIVE: Real money routing (requires ENABLE_LIVE=true)
  - PAPER: Simulated fills with realistic slippage
  - SHADOW: Parallel tracking without execution

### Notifications & Observability
- **Telegram integration**:
  - Enhanced notifier with multi-chat support (ALLOWED_CHAT_IDS)
  - Webhook mode for cloud deployments (push-based, low latency)
  - Polling fallback for development/testing
  - Command interface: /tick, /ws_status, /execstate, /emergency_stop
- **Structured logging**: JSON-formatted logs with correlation IDs for traceability
- **Prometheus metrics**: 50+ exported metrics (latency, fill rates, PnL, error counters)
- **Health endpoint**: /health with degraded-mode detection
- **Diagnostic snapshots**: YAML-based state dumps for post-mortem analysis

## Project Structure

```
src/nifty_scalper_bot/
├── main.py                 # FastAPI entrypoint with lifespan management
├── core/
│   ├── app.py              # Main orchestrator (NiftyScalperApp)
│   ├── strategy_manager.py # Strategy registration & lifecycle
│   ├── market_regime.py    # Regime detection HMM + heuristics
│   ├── unified_manager.py  # Cross-component coordination
│   └── option_universe.py  # Option chain filtering & selection
├── strategies/
│   ├── runner.py           # Strategy execution harness
│   ├── elite_strategies/   # Production alpha models
│   ├── signal_generator.py # Micro-structure signal generation
│   ├── indicators.py       # Technical indicator library
│   └── base_strategy.py    # Abstract strategy interface
├── execution/
│   ├── lifecycle_manager.py    # TP/SL/trailing logic
│   ├── order_manager.py        # Order creation & modification
│   ├── position_manager.py     # Active position tracking
│   ├── bracket_manager.py      # Bracket order orchestration
│   └── shadow_paper.py         # Paper trading simulator
├── risk/
│   ├── risk_manager.py     # Portfolio-level risk aggregation
│   ├── session_gate.py     # Market hours enforcement
│   ├── position_sizing.py  # Dynamic sizing algorithms
│   └── volatility_sizer.py # ATR-based position scaling
├── data/
│   ├── data_hub.py         # Centralized market data access
│   ├── instruments.py      # Token resolution & caching
│   ├── market_data_manager.py # OHLCV aggregation
│   ├── persistent_state.py # SQLite state persistence
│   └── rest/zerodha_client.py # Broker REST API wrapper
├── streaming/
│   ├── websocket_manager.py    # Kite WS protocol handler
│   ├── polling_streamer.py     # REST-based fallback streamer
│   ├── resilient_streamer.py   # Auto-reconnect with backoff
│   └── stream_supervisor.py    # Health monitoring & failover
├── notifications/
│   ├── telegram_webhook_enhanced.py # Push notification server
│   ├── telegram_commands.py         # Slash command handlers
│   └── telegram_service.py          # Message dispatch
├── infra/
│   ├── metrics.py          # Prometheus exporters
│   ├── health.py           # Health check implementation
│   ├── watchdog.py         # Process liveness monitoring
│   └── scheduled_tasks.py  # Cron-like background jobs
├── config/
│   ├── base.py             # Pydantic configuration models
│   ├── settings.py         # Environment variable loading
│   └── paths.py            # Data directory resolution
└── utils/
    ├── rate_limiter.py     # Token bucket implementation
    ├── circuit_breaker.py  # Failure isolation
    ├── logging.py          # Structured logger factory
    └── pricing.py          # Options pricing utilities
```

## Getting Started

### Prerequisites
- Python 3.10+
- Zerodha Kite Connect account (or dummy broker for testing)
- Telegram Bot Token (optional, for alerts)

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Configure environment**:

   Create `.env` file with required variables:
   ```bash
   # Broker credentials (Zerodha)
   export ZERODHA_API_KEY=your_api_key
   export ZERODHA_ACCESS_TOKEN=your_access_token
   export ZERODHA_WS_ORIGIN=https://kite.zerodha.com

   # Execution mode
   export ENABLE_LIVE=false          # Set to 'true' for real trading
   export EXECUTION_MODE=SHADOW      # LIVE | PAPER | SHADOW

   # Telegram notifications
   export TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   export TELEGRAM_CHAT_ID=987654321

   # Risk parameters
   export RISK_MAX_DAILY_TRADES=20
   export RISK_MAX_ORDER_NOTIONAL=200000
   export RISK_PER_TRADE_CAP_PCT=1.0
   ```

4. **Run the bot**:
   ```bash
   # Development mode (dummy broker)
   python -m nifty_scalper_bot.app

   # Production mode (FastAPI server)
   uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port 8000
   ```

## Configuration Reference

### Critical Environment Variables

| Category | Variable | Description | Default |
|----------|----------|-------------|---------|
| **Broker** | ZERODHA_API_KEY | Kite Connect API key | *required* |
| | ZERODHA_ACCESS_TOKEN | Session access token | *required* |
| | ZERODHA_WS_ORIGIN | WebSocket origin header | https://kite.zerodha.com |
| **Execution** | ENABLE_LIVE | Master switch for live orders | false |
| | EXECUTION_MODE | Routing mode | SHADOW |
| **Risk** | RISK_MAX_DAILY_TRADES | Maximum trades per day | 20 |
| | RISK_MAX_ORDER_NOTIONAL | Max order value (INR) | 200000 |
| | RISK_PER_TRADE_CAP_PCT | Capital allocation per trade | 1.0 |
| **Lifecycle** | LIFECYCLE_TP1_R | TP1 risk-reward multiple | 1.0 |
| | LIFECYCLE_TP1_PARTIAL | Fraction exited at TP1 | 0.6 |
| | LIFECYCLE_TP2_R_TREND | TP2 multiple in TREND regime | 1.8 |
| | LIFECYCLE_TP2_R_RANGE | TP2 multiple in RANGE regime | 1.4 |
| | LIFECYCLE_TRAIL_ATR_MULT | ATR multiplier for trailing | 0.8 |
| | LIFECYCLE_TIME_STOP_MIN | Auto-exit after minutes | 12 |
| **Streaming** | STREAM__MODE | Data transport mode | websocket |
| | POLL_INTERVAL_MS | REST polling interval | 700 |
| | POLL_BATCH_SIZE | Tokens per batch request | 200 |

## Operational Commands (Telegram)

| Command | Description |
|---------|-------------|
| `/tick <token>` | Real-time quote for instrument |
| `/ws_status` | WebSocket connection health |
| `/execstate <symbol>` | Position lifecycle state |
| `/execqueue` | Pending order queue |
| `/execlast` | Recent execution decisions |
| `/execwhy <symbol>` | Preflight gate diagnostics |
| `/emergency_stop` | Immediate position square-off |
| `/pause_trading` | Halt new order intake |
| `/resume_trading` | Resume order processing |
| `/pnl` | Realized/unrealized PnL summary |
| `/regime` | Current market regime |

## Deployment Guide

### Railway/Cloud Deployment

**Recommended configuration for ephemeral environments**:

```bash
# Force polling mode (WebSocket unreliable on dynamic IPs)
export WEBSOCKET__DISABLED=true
export STREAM__MODE=poll
export POLL_INTERVAL_MS=700
export POLL_BATCH_SIZE=50

# Disable webhook (use polling for Telegram)
export TELEGRAM__WEBHOOK_ENABLED=false
```

**Environment variables required**:
- ZERODHA_API_KEY
- ZERODHA_ACCESS_TOKEN
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- ENABLE_LIVE (set carefully!)

**Deployment steps**:
1. Push code to Git-connected Railway project
2. Configure all environment variables in Railway dashboard
3. Check `/health` endpoint returns `{"status": "running", "bot_loaded": true}`
4. Test Telegram commands: `/tick 256265` (NIFTY spot token)

### Troubleshooting WebSocket Issues

**Error 1006 (Abnormal Closure)**:
- **Cause 1**: Expired access token - Regenerate via login flow
- **Cause 2**: Multiple concurrent sockets - Terminate other instances (one socket per token)
- **Cause 3**: Network instability - Switch to polling mode temporarily

**Verification**:
```bash
curl -H "Authorization: token <API_KEY>:<ACCESS_TOKEN>" \
  https://api.kite.trade/user/profile
```

## Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests (Paper Trading)
```bash
export EXECUTION_MODE=PAPER
export SESSION_ALLOW_OUT_OF_HOURS=true
pytest tests/integration/ -v
```

### Tick Replay Backtesting
```bash
python -m nifty_scalper_bot.testing.run_tick_replay \
  --date 2024-12-10 \
  --strategy elite_orb \
  --initial-capital 500000
```

## Diagnostics & Debugging

### Key Log Patterns
- `preflight_block_*`: Order rejected by risk gate
- `shadow_drift_bps`: Simulated vs actual fill divergence
- `reconcile_mismatch`: Internal state != broker snapshot
- `regime_change`: Market regime transition detected
- `lifecycle_exit_*`: Position closed (TP/SL/trailing/time-stop)

### Metric Dashboards (Prometheus)
- `orders_submitted_total{mode="LIVE"}`: Live order count
- `order_fill_latency_seconds`: Time from submission to acknowledgment
- `position_pnl_unrealized`: Current open PnL
- `stream_tick_lag_ms`: WebSocket/polling latency
- `risk_circuit_breaker_trips`: Safety mechanism activations

## Risk Warnings

1. **Never enable ENABLE_LIVE=true without thorough paper-trading validation**
2. **Monitor shadow drift daily** - Persistent >20bps drift indicates model-broker mismatch
3. **Respect rate limits** - Exceeding broker API limits triggers temporary bans
4. **Test emergency procedures** - Regularly practice `/emergency_stop` in paper mode
5. **Review reconciliation logs** - Unresolved mismatches may indicate missed fills

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feat/awesome-strategy`)
3. Run type checking: `mypy src/nifty_scalper_bot`
4. Ensure test coverage: `pytest --cov=nifty_scalper_bot`
5. Submit PR with detailed description of changes

---

**Built for systematic traders** - [Documentation](docs/) - [Production Playbook](docs/production_playbook.md)


## Current execution architecture

StrategyRunner → OrderManager → BracketManager

- StrategyRunner builds and submits TradePlan objects.
- OrderManager owns entry order placement to broker/paper execution.
- BracketManager owns SL/TP/trailing/EOD virtual bracket exits.

## Paper/Shadow single-vote tuning (optional)
For paper or shadow validation only (keep live defaults conservative):

```env
STRATEGY_ALLOW_SINGLE_VOTE_SCALP=true
STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE=5.5
STRATEGY_SINGLE_VOTE_VWAP_MIN_CONFIDENCE=0.45
```

Live defaults remain stricter (`STRATEGY_ALLOW_SINGLE_VOTE_SCALP=false`, `STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE=5.8`).

## Startup Hydration Sequence

The live startup gate uses one hydration contract (`HydrationStatus`) so every
layer reports the same symbol, role, token, quote, depth, and bar-count state.
Startup must not arm live orders until the selected CE/PE have both evaluation
and execution history.

```text
InstrumentManager resolves NIFTY spot, active future, selected CE/PE, nearby options
→ MarketDataManager registers symbol/token maps and subscribes WebSocket tokens
→ MarketDataManager fetches historical OHLC outside the tick path
→ MarketDataManager ingests sorted UTC bars and merges live candles on top
→ DataHub reads the MDM OHLC/quote cache
→ StrategyRunner reseeds runner history from DataHub/MDM bars
→ IndicatorEngine receives the same reseeded bars
→ execution readiness checks quote/depth/spread + required execution bars
→ live_orders_armed=true only after broker health + hydration + risk gates pass
```

Expected concise log sequence:

```text
ACTIVE_BASKET_INITIAL_HYDRATION_PENDING pending_ce=NFO:...CE pending_pe=NFO:...PE ce_bars=1 pe_bars=1 required=30
HYDRATION_FETCH_ATTEMPT symbol=NFO:...CE token=... tradingsymbol=... interval=minute attempt=token
HYDRATION_FETCH_RESULT symbol=NFO:...CE returned_rows=30 accepted_rows=30 first_ts=... last_ts=...
HYDRATION_INGEST_RESULT symbol=NFO:...CE returned_rows=30 accepted_rows=29 final_mdm_bars=30
RUNNER_HISTORY_RESEEDED symbol=NFO:...CE runner_bars=30 indicator_bars=30 min_bars=30 source=selected_option_history_prewarm
ACTIVE_BASKET_PROMOTED selected_ce=NFO:...CE selected_pe=NFO:...PE ce_bars=30 pe_bars=30 required=30
READINESS_BLOCKER_SUMMARY blockers=[] data_hard_ready=True evaluation_ready=True execution_ready=True live_orders_armed=True
```

## StrategyRunner same-bar evaluation throttle

StrategyRunner can skip repeated same-bar strategy computation while still
letting hydration, DataHub updates, active basket selection, subscriptions, and
readiness state proceed normally. New bars evaluate immediately; same-bar
periodic re-evaluation is allowed after this interval, with a runtime lower bound
of 3 seconds.

```env
RUNNER_SAME_BAR_PERIODIC_EVAL_SECONDS=5
```
