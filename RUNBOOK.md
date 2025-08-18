# RUNBOOK.md: Nifty 50 Scalper Bot Operator's Guide

This document provides instructions for developers and operators on how to set up, run, and troubleshoot the Nifty 50 Scalper Bot.

---

## 1. Local Development Setup

Follow these steps to set up a local development environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/kundakarlabp/nifty_scalper_bot.git
    cd nifty_scalper_bot
    ```

2.  **Create a Virtual Environment**
    This project uses Python 3.11+.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Configuration

The application is configured using environment variables. For local development, you can create a `.env` file in the root of the repository.

1.  **Create the `.env` File**
    Copy the example file to create your own local configuration.
    ```bash
    cp .env.example .env
    ```
    *(Note: `.env.example` will be created during the refactoring process. For now, you can create an empty `.env` file and add the variables below.)*

2.  **Populate Required Variables**
    You will need to provide credentials for Zerodha and Telegram.

    ```ini
    # .env

    # Zerodha API Credentials
    ZERODHA_API_KEY="YOUR_API_KEY"
    ZERODHA_API_SECRET="YOUR_API_SECRET"
    ZERODHA_ACCESS_TOKEN="YOUR_ACCESS_TOKEN" # This is a short-lived token

    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"

    # --- Trading Mode (IMPORTANT) ---
    # Set to "true" to enable live order placement. Default is "false" (shadow mode).
    ENABLE_LIVE_TRADING="false"
    ```

---

## 3. Running the Bot

The application has two primary modes: live/shadow trading and backtesting.

### Live / Shadow Trading

Use the main entry point to start the bot. It will connect to Zerodha, subscribe to market data, and begin monitoring for trading signals.

```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Start the bot
python3 -m src.main start
```

-   If `ENABLE_LIVE_TRADING` is `false` (default), the bot runs in **shadow mode**. It will generate signals and log them, but no actual orders will be placed.
-   If `ENABLE_LIVE_TRADING` is `true`, the bot will execute live trades. **Use with caution.**

### Backtesting

The backtest engine allows you to test the strategy on historical data.

```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Run the backtest
python3 tests/true_backtest_dynamic.py
```

The backtester will run using the data specified within the script (e.g., from `src/data/nifty_ohlc.csv`) and will output a detailed performance report to the console and to the `/reports` directory.

---

## 4. Deployment (Render / Railway)

This section will be updated with detailed instructions for cloud deployment once the refactoring is complete. The general process will be:

1.  Create a new "Web Service" on the platform.
2.  Connect it to your forked GitHub repository.
3.  Set the environment variables from your `.env` file in the platform's "Secrets" or "Environment Variables" section.
4.  Use the `Dockerfile` provided in the repository. The platform should automatically detect and use it.
5.  The start command will be automatically configured from the `Procfile` or `render.yaml`.

---

## 5. Troubleshooting

-   **Authentication Errors**: Ensure your `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, and `ZERODHA_ACCESS_TOKEN` are correct and that the access token has not expired.
-   **Telegram Errors**: Verify your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
-   **Dependency Issues**: If you encounter errors after a `git pull`, re-run `pip install -r requirements.txt` to ensure your dependencies are up to date.
