# Nifty 50 Scalper Bot

An automated trading bot for Nifty 50 options, designed to run on [Render](https://render.com) using the Zerodha KiteConnect API. The bot implements a scalping strategy with adaptive signal scoring, capital preservation rules, and Telegram-based control.

## Features

*   **Zerodha KiteConnect Integration:** Connects to live market data and executes trades.
*   **Adaptive Scalping Strategy:** Uses multi-indicator scoring (EMA, RSI, MACD, etc.) and market regime detection.
*   **Risk Management:** Implements fixed risk per trade, daily loss limits, and dynamic position sizing.
*   **Telegram Control:** Start, stop, pause, and monitor the bot via Telegram commands.
*   **Dynamic Trailing Stop-Loss:** Manages Stop-Loss and Take-Profit using Kite's GTT orders, with simulated trailing logic.
*   **Automatic Instrument Selection:** Selects ATM/OTM Nifty 50 options based on the next expiry.
*   **Modular Design:** Well-organized codebase for easy maintenance and extension.

## Prerequisites

*   Python 3.8+
*   A Zerodha account with Kite API access (API Key, API Secret)
*   A Telegram Bot Token (create via BotFather) and your Telegram User ID
*   Docker (for local testing/building or Render deployment)
*   A Render account (for deployment)

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd nifty_scalper_bot
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your actual credentials:
        ```
        # .env
        ZERODHA_API_KEY=your_actual_api_key_here
        ZERODHA_API_SECRET=your_actual_api_secret_here
        KITE_ACCESS_TOKEN=your_initial_access_token_here # Can be generated via auth script
        TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
        TELEGRAM_USER_ID=your_telegram_user_id_here

        # Optional Config (defaults are in config.py)
        # ACCOUNT_SIZE=100000.0
        # RISK_PER_TRADE=0.01
        # MAX_DRAWDOWN=0.05
        # CONFIDENCE_THRESHOLD=8.0
        # BASE_STOP_LOSS_POINTS=20.0
        # BASE_TARGET_POINTS=40.0
        # DEFAULT_PRODUCT=MIS
        # DEFAULT_ORDER_TYPE=MARKET
        # DEFAULT_VALIDITY=DAY
        ```

4.  **(Optional) Generate Initial Access Token:**
    If you don't have a `KITE_ACCESS_TOKEN`, you might need to run an authentication script (like the one potentially in `src/auth/zerodha_auth.py`) to generate one and save it to your `.env` file.

## Running the Bot Locally

*   **Start Trading:**
    ```bash
    python src/main.py --mode realtime --trade
    ```
*   **Signal Generation Only (No Execution):**
    ```bash
    python src/main.py --mode realtime
    ```
*   **Test Telegram Alerts:**
    ```bash
    python test_telegram.py
    ```

## Deployment on Render

1.  Ensure your `.env` variables are set in the Render dashboard (Environment Variables section) instead of using a local `.env` file.
2.  Push your code to the repository connected to your Render service.
3.  Render will automatically build and deploy using the `Dockerfile` and `render.yaml`.

## Configuration (`config.py`)

Central configuration file for strategy parameters, risk settings, and instrument details. Modify defaults here or override with `.env` variables.

## Logging

Logs are written to `logs/trading_bot.log` with rotation (max 5 files, 10MB each) and also output to the console.

## Project Structure
nifty_scalper_bot/
├── config.py
├── .env
├── .env.example
├── logs/
├── requirements.txt
├── Dockerfile
├── render.yaml
├── manage_bot.sh
├── test_telegram.py
├── README.md
├── src/
│ ├── main.py
│ ├── auth/
│ │ └── zerodha_auth.py
│ ├── data_streaming/
│ │ ├── market_data_streamer.py
│ │ ├── data_processor.py
│ │ └── realtime_trader.py
│ ├── execution/
│ │ └── order_executor.py
│ ├── strategies/
│ │ └── scalping_strategy.py
│ ├── risk/
│ │ └── position_sizing.py
│ ├── notifications/
│ │ └── telegram_controller.py
│ ├── utils/
│ │ ├── strike_selector.py
│ │ └── expiry_selector.py
│ └── backtesting/ (Planned)
└── tests/ (Planned)


## Future Enhancements

*   Implement backtesting module.
*   Add more sophisticated state persistence (e.g., SQLite, Redis).
*   Integrate Kite's OMS WebSocket for real-time order updates.
*   Add more unit and integration tests.
