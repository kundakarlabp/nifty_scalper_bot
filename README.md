# Nifty Scalper Bot

This repository contains an advanced options scalping bot designed for the Nifty‑50 index.  It integrates with Zerodha's KiteConnect API for market data and order placement, and exposes a Telegram interface for control and monitoring.  The codebase has been refactored for clarity, stability and extensibility.  Key features include:

* **Modular architecture** – core components are separated into strategy, risk management, execution and notifications.
* **Indicator‑rich strategy** – combines EMA, RSI, MACD, ATR, SuperTrend, VWAP, ADX and Bollinger bandwidth to generate high‑confidence signals.  Market regime detection adjusts behaviour for trending versus ranging conditions.
* **Adaptive position sizing** – calculates lot sizes based on account capital, risk per trade, market volatility, recent performance and daily drawdown limits.
* **Good‑Till‑Triggered (GTT) orders with trailing logic** – entries automatically attach stop‑loss and take‑profit orders and adjust them when the trade moves in your favour.
* **Telegram bot control** – supports `/start`, `/stop`, `/status` and `/summary` commands.  It pushes P&L updates, error alerts and session notifications to the configured chat.
* **Dockerised deployment** – a minimal Dockerfile and Railway/Render configuration make it easy to run the bot in a cloud environment.

nifty_scalper_bot/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── auth/
│   ├── backtesting/
│   ├── data/
│   ├── data_streaming/
│   │   ├── __init__.py
│   │   └── realtime_trader.py
│   ├── execution/
│   │   ├── __init__.py
│   │   └── order_executor.py
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── telegram_controller.py
│   ├── risk/
│   │   ├── __init__.py
│   │   └── position_sizing.py
│   ├── scripts/
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── scalping_strategy.py
│   ├── utils/
│   └── config.py
├── Dockerfile
├── manage_bot.sh
├── render.yaml
├── requirements.txt
└── ...
## Setup & Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/kundakarlabp/nifty_scalper_bot.git
   cd nifty_scalper_bot
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** based on `.env.example` and populate the required fields.  At a minimum you will need your Zerodha API key/secret, a short‑lived access token and your Telegram bot token/ID.

4. **Run the bot locally**

   ```bash
   # sanity + start
   ./manage_bot.sh run
   ```

   The wrapper validates dependencies then launches `python -m src.main start`.  Logs stream to the console and rotate under `logs/`.  Use the Telegram commands documented below to control the session.

5. **Deploy to Railway/Render**

   The provided `Dockerfile` and `render.yaml` work with Render or Railway.  Create a new Web Service in your chosen platform, point it at this repository and add the environment variables specified in `.env.example` via the dashboard.  The default command will start the bot and begin listening for Telegram commands.

## Architecture & Docker

The runtime wiring is broker‑agnostic:

```
instruments → KiteBroker → BrokerDataSource → Orchestrator → BrokerOrderExecutor
```

```
                 +----------------+
Market data ---> | BrokerDataSource| --+
                 +----------------+   |
                                       v
                               +---------------+
                               | StrategyRunner| --+--> OrderExecutor --> Broker API
                               +---------------+   |
                                       ^            |
                                       |            v
                                  Health server  Telegram
                                       ^            |
                                       +------------+
```

`src/main.py` assembles these pieces and optionally attaches a Telegram notifier if credentials are provided.  A minimal Docker setup is provided for local runs:

```bash
cp .env.example .env  # fill in KITE_* and SYMBOL
docker compose up --build bot      # run the bot
docker compose up --build test     # run the test suite
```

## Health endpoints

A lightweight Flask app exposes health probes on port `8000` by default:

```
GET  /health  → JSON status bundle
HEAD /health  → cheap probe for platforms
```

Disable the server by setting `HEALTH__ENABLE_SERVER=false` in the environment.

## Risk flags & kill switch

Risk checks guard every trade.  The runner tracks:

- max trades per day
- consecutive loss limit
- daily drawdown percentage
- equity floor

Use the Telegram `/risk` command to inspect current limits.  Live trading is disabled unless `ENABLE_LIVE_TRADING=true` (alias `ENABLE_TRADING=true`); setting the flag to false acts as a kill switch.

## Telegram control

The bot can react to simple commands sent to the configured chat:

* `/start` – begin a real‑time trading session.  The bot will connect to Kite, subscribe to market data and start evaluating signals.
* `/stop` – halt streaming and cancel any open subscriptions.  Existing positions will continue to be managed by the execution layer.
* `/status` – return a summary of the current trading state including uptime, streaming status, active positions, instruments and risk metrics.
* `/summary` – report realised P&L for the day, the number of trades taken and the current drawdown.
* `/pause` – temporarily stop processing ticks and generating new orders.
* `/resume` – resume processing after a pause.

## Troubleshooting

- **Health check failing** – ensure port `8000` is free and the process has started.  `curl localhost:8000/health` should return JSON.
- **Risk block** – `/status` or `/risk` will show which gate triggered.  Reset with `/riskresettoday` if appropriate.
- **Network down or stale ticks** – check connectivity to the broker and restart `manage_bot.sh run` once connectivity is restored.
- **Broker order rejected** – review logs and `/apihealth`; common causes are invalid parameters or rate limits.

## Testing

Basic unit tests live under `tests/`.  They exercise the strategy scoring logic and the position sizing module with synthetic data.  To run the test suite:

```bash
pytest -q
```

To run the same checks as CI, install and execute the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Contributing

Pull requests are welcome.  Please open an issue to discuss major changes before submitting a patch.  When updating the strategy or risk modules, include corresponding tests demonstrating the new behaviour.
