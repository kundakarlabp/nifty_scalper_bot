# Nifty Scalper Bot

This repository contains an advanced options scalping bot designed for the Nifty‑50 index.  It integrates with Zerodha's KiteConnect API for market data and order placement, and exposes a Telegram interface for control and monitoring.  The codebase has been refactored for clarity, stability and extensibility.  Key features include:

* **Modular architecture** – core components are separated into strategy, risk management, execution and notifications.
* **Indicator‑rich strategy** – combines EMA, RSI, MACD, ATR, SuperTrend, VWAP, ADX and Bollinger Band width to generate high‑confidence signals.  Market regime detection adjusts behaviour for trending versus ranging conditions.
* **Adaptive position sizing** – calculates lot sizes based on account capital, risk per trade, market volatility, recent performance and daily drawdown limits.
* **Good‑Till‑Triggered (GTT) orders with trailing logic** – entries automatically attach stop‑loss and take‑profit orders and adjust them when the trade moves in your favour.
* **Telegram bot control** – supports `/start`, `/stop`, `/status` and `/summary` commands.  It pushes P&L updates, error alerts and session notifications to the configured chat.
* **Dockerised deployment** – a minimal Dockerfile and Railway/Render configuration make it easy to run the bot in a cloud environment.

nifty_scalper_bot/
├── src/                      # All core bot logic lives here
│   ├── main.py               # ✅ Entry point
│   ├── config.py             # ✅ Central configuration file (API keys, SL/TP, etc.)
│   ├── data_streaming/       # ✅ Market data & streaming logic
│   ├── execution/            # ✅ Order execution logic
│   ├── notifications/        # ✅ Telegram bot control & alerts
│   ├── risk/                 # ✅ Position sizing, daily limits
│   ├── strategies/           # ✅ Technical indicator logic
│   └── utils/                # ✅ Utility tools: strike selector, expiry, etc.
│
├── tests/                    # 🔧 (Optional) Test scripts
│
├── .env.example              # ✅ Sample environment variable file
├── Dockerfile                # ✅ For containerization
├── manage_bot.sh             # ✅ Shell script to run/kill the bot
├── render.yaml               # ✅ For Render or Railway deployment config
├── README.md                 # ✅ Documentation
├── requirements.txt          # ✅ Python dependencies

## Quick start

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
   python3 -m src.main
   ```

   The bot will log to both the console and a rotating file under `logs/`.  Use the Telegram commands documented below to control the session.

5. **Deploy to Railway/Render**

   The provided `Dockerfile` and `render.yaml` work with Render or Railway.  Create a new Web Service in your chosen platform, point it at this repository and add the environment variables specified in `.env.example` via the dashboard.  The default command will start the bot and begin listening for Telegram commands.

## Telegram commands

* `/start` – begin a real‑time trading session.  The bot will connect to Kite, subscribe to market data and start evaluating signals.
* `/stop` – halt streaming and cancel any open subscriptions.  Existing positions will continue to be managed by the execution layer.
* `/status` – return a summary of the current trading state including uptime, streaming status, active positions, instruments and risk metrics.
* `/summary` – report realised P&L for the day, the number of trades taken and the current drawdown.

## Testing

Basic unit tests live under `tests/`.  They exercise the strategy scoring logic and the position sizing module with synthetic data.  To run the test suite:

```bash
pytest -q
```

## Contributing

Pull requests are welcome.  Please open an issue to discuss major changes before submitting a patch.  When updating the strategy or risk modules, include corresponding tests demonstrating the new behaviour.
