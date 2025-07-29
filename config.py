# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
load_dotenv(dotenv_path=Path('.') / '.env')

class Config:
    # Zerodha Credentials
    ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY")
    ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

    # Telegram Bot
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

    # Risk Settings
    MAX_RISK_PER_TRADE = 0.01   # 1% of capital
    DAILY_MAX_LOSS = 0.05       # 5%
    CONSECUTIVE_LOSS_LIMIT = 3

    # Trade Filters
    MIN_SIGNAL_SCORE = 7.0
    TIME_FILTER_START = "09:15"
    TIME_FILTER_END = "15:15"

    # Paths
    LOG_FILE = "logs/trades.csv"