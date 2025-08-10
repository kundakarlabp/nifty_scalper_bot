# src/config.py
"""
Global configuration for Nifty Scalper Bot.

All settings can be overridden via environment variables (.env).
"""

import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class Config:
    # ======================
    # API Keys / Tokens
    # ======================
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_ACCESS_TOKEN: str = os.getenv("ZERODHA_ACCESS_TOKEN", "")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ======================
    # Trading Settings
    # ======================
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ALLOW_OFFHOURS_TESTING: bool = os.getenv("ALLOW_OFFHOURS_TESTING", "false").lower() in ("true", "1", "yes")
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", "5"))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", "10"))
    CONFIDENCE_THRESHOLD: int = int(os.getenv("CONFIDENCE_THRESHOLD", "8"))
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "2"))
    CAPITAL_PER_TRADE: float = float(os.getenv("CAPITAL_PER_TRADE", "30000"))
    RISK_PER_TRADE_PERCENT: float = float(os.getenv("RISK_PER_TRADE_PERCENT", "1"))

    # ======================
    # Risk & Position Sizing
    # ======================
    MIN_POSITION_QTY: int = int(os.getenv("MIN_POSITION_QTY", "50"))
    MAX_POSITION_QTY: int = int(os.getenv("MAX_POSITION_QTY", "900"))
    POSITION_QTY_STEP: int = int(os.getenv("POSITION_QTY_STEP", "50"))

    # ======================
    # Execution Defaults
    # ======================
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")
    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", "1.0"))
    TRAIL_COOLDOWN_SEC: int = int(os.getenv("TRAIL_COOLDOWN_SEC", "15"))
    TICK_SIZE: float = float(os.getenv("TICK_SIZE", "0.05"))
    PREFERRED_EXIT_MODE: str = os.getenv("PREFERRED_EXIT_MODE", "AUTO").upper()  # AUTO / GTT / REGULAR

    # ======================
    # Strategy Parameters
    # ======================
    EMA_FAST_PERIOD: int = int(os.getenv("EMA_FAST_PERIOD", "9"))
    EMA_SLOW_PERIOD: int = int(os.getenv("EMA_SLOW_PERIOD", "21"))
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    BB_WINDOW: int = int(os.getenv("BB_WINDOW", "20"))
    BB_STD: int = int(os.getenv("BB_STD", "2"))
    STOCH_K_PERIOD: int = int(os.getenv("STOCH_K_PERIOD", "14"))
    STOCH_D_PERIOD: int = int(os.getenv("STOCH_D_PERIOD", "3"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    SCORING_THRESHOLD: int = int(os.getenv("SCORING_THRESHOLD", "9"))

    # ======================
    # Logging / Monitoring
    # ======================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    BALANCE_LOG_INTERVAL_MIN: int = int(os.getenv("BALANCE_LOG_INTERVAL_MIN", "30"))

    # ======================
    # Scheduling
    # ======================
    FETCH_INTERVAL_SEC: int = int(os.getenv("FETCH_INTERVAL_SEC", "30"))

    # ======================
    # Strike Selector
    # ======================
    STRIKE_STEP: int = int(os.getenv("STRIKE_STEP", "50"))

    # ======================
    # Telegram Commands Control
    # ======================
    TELEGRAM_COMMAND_PREFIX: str = os.getenv("TELEGRAM_COMMAND_PREFIX", "/")