
📈 Nifty Scalper Bot

An advanced Nifty-50 options scalping bot built for production trading.
It integrates with Zerodha KiteConnect API for live market data and order execution, and with Telegram Bot API for remote control and monitoring.

The system is designed to be modular, fault-tolerant, adaptive, and risk-aware, with configurable strategies and safety mechanisms to support live deployment on cloud platforms (Railway, Render, etc.).


---

🚀 Key Features

Modular architecture

strategies/ → Indicator-rich scalping logic with regime detection

execution/ → Order placement, stop-loss/target handling, retry & safety

risk/ → Position sizing, circuit breakers, drawdown protection

notifications/ → Telegram control + alerts

data_streaming/ → Real-time market data streaming & orchestration


Indicator-rich signal engine

EMA, RSI, MACD, ATR, SuperTrend, VWAP, ADX, Bollinger Bands

Multi-timeframe filters (e.g., 1-min execution, 5-min trend check)

Market regime filters (Trending vs Ranging)


Adaptive position sizing

Risk per trade, capital-based sizing, ATR volatility adjustment

Daily risk controls, loss streak halving, circuit breaker


Smart execution

Market/GTT orders with linked SL/TP

ATR-based dynamic targets & stops

Trailing SL, breakeven logic, partial profit booking


Telegram control interface

Start/Stop trading sessions

Switch between LIVE / SHADOW (paper) modes

Toggle Quality filter (On/Off/Auto)

Set regime gates (Auto/Trend/Range)

Risk adjustment & pause/resume

Status, P&L summary, health checks


Production deployment ready

Railway/Render worker profile with manage_bot.sh

.env driven configuration (no secrets in repo)

Logging to console + CSV + optional pinned Telegram status




---

📂 Project Structure

nifty_scalper_bot/
├── src/
│   ├── main.py                 # Entry point
│   ├── auth/                   # Zerodha authentication (KiteConnect)
│   ├── backtesting/            # Backtest runner + data utils
│   ├── data_streaming/
│   │   └── realtime_trader.py  # Live trader orchestration
│   ├── execution/
│   │   └── order_executor.py   # Order placement, exits, retries
│   ├── notifications/
│   │   └── telegram_controller.py # Telegram command interface
│   ├── risk/
│   │   └── position_sizing.py  # Lot sizing, drawdown protection
│   ├── strategies/
│   │   └── scalping_strategy.py# Signal scoring engine
│   ├── utils/                  # Strike selection, helpers
│   └── config.py               # Config loader (from .env)
├── manage_bot.sh               # Supervisor script
├── Dockerfile                  # Cloud deployment
├── render.yaml / railway.toml  # Platform configs
├── requirements.txt
└── logs/                       # Trade logs (runtime)


---

⚙️ Configuration (.env)

The bot is entirely configurable via .env.

Core Toggles

ENABLE_LIVE_TRADING=true      # true=real trades, false=shadow
ENABLE_TELEGRAM=true
ALLOW_OFFHOURS_TESTING=false
USE_IST_CLOCK=true
SESSION_AUTO_EXIT_TIME=15:30  # auto exit at market close

Zerodha & Telegram

ZERODHA_API_KEY=xxxx
KITE_ACCESS_TOKEN=xxxx        # refreshed daily
TELEGRAM_BOT_TOKEN=xxxx
TELEGRAM_CHAT_ID=xxxx

Strategy Core

MIN_SIGNAL_SCORE=2
CONFIDENCE_THRESHOLD=5.2
BASE_STOP_LOSS_POINTS=20.0
BASE_TARGET_POINTS=40.0

Risk Management

RISK_PER_TRADE=0.025          # 2.5% risk per trade
MAX_DRAWDOWN=0.07
CONSECUTIVE_LOSS_LIMIT=3
MAX_DAILY_DRAWDOWN_PCT=0.05

Polling Cadence

POLL_SEC=5                    # base poll interval (seconds)

(This replaces older PEAK_POLL_SEC / OFFPEAK_POLL_SEC logic.)


---

📱 Telegram Commands

Command	Function

/start	Begin trading session
/stop	Stop trading & exit gracefully
`/mode live	shadow`
`/quality on	off
`/regime auto	trend
/risk 0.5	Adjust risk per trade (%)
/pause 1	Pause new entries (min)
/resume	Resume after pause
/status	Show bot status (mode, quality, PnL, uptime)
/summary	Daily summary of trades & P&L
/refresh	Reload balance/instruments
/health	System health check
/emergency	Exit all positions & cancel orders



---

🛡️ Safety Mechanisms

Drawdown guards – shuts down after defined % loss

Loss streak halving – reduces position size after consecutive losses

Daily trade limits – stops after max trades per day

Spread guard – blocks trades in wide bid/ask spreads

Circuit breaker – cool-off after sharp losses



---

🖥️ Deployment

Railway

Profile contains worker definition:

worker: bash manage_bot.sh run

.env variables should be provided via file in repo (preferred) or via dashboard.


Local

bash manage_bot.sh run   # starts bot (default shadow mode)
bash manage_bot.sh start # explicit start
bash manage_bot.sh stop  # graceful stop
bash manage_bot.sh status# status


---

📊 Logs

Trades are written to logs/trades.csv

Balance and P&L snapshots logged at intervals

Telegram receives real-time trade alerts



---

🔮 Future Enhancements

ML-based trade classification (RandomForest/XGBoost)

Option chain OI filters & DOM depth tracking

Smarter time-of-day filters & event blackout windows

Auto-adaptive risk budget & regime optimization
