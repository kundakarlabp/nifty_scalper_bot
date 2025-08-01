"""
Centralised configuration loader for the scalper bot.  Reads environment
variables at import time and exposes them via the ``Config`` class.  All
credentials and tuning parameters should be defined in a ``.env`` file at
the project root (see ``.env.example``) or provided via the runtime
environment.
"""
import os
from pathlib import Path
# Attempt to import ``load_dotenv``.  If the ``pythonâ€‘dotenv`` package is
# unavailable the import will fail silently and environment variables
# will not be loaded from a .env file.  This fallback ensures that
# absence of the optional dependency does not break the application.
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(*args, **kwargs) -> None:  # type: ignore
        """Fallback noâ€‘op for load_dotenv when pythonâ€‘dotenv is missing."""
        return None

# Load .env file if present
load_dotenv(dotenv_path=Path('.') / '.env')


class Config:
    """
    Static configuration values pulled from environment variables.  Default
    values are provided for most fields to allow local development without
    a .env file.  See ``.env.example`` for a full list of supported keys.
    """
    # Zerodha credentials
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID = 0

    # Risk and money management
    ACCOUNT_SIZE: float = float(os.getenv("ACCOUNT_SIZE", 100_000.0))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% of capital per trade
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", 0.05))      # 5% daily drawdown limit
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # Strategy tuning
    # Base stop loss and target distances in points.  These serve as a
    # minimum when calculating adaptive stopâ€‘losses and takeâ€‘profits.  In
    # practice the final values are derived from ATR and other factors in
    # ``EnhancedScalpingStrategy``.
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    # Minimum confidence required (0â€“10 scale) for a signal to be acted upon.
    # This value is converted to an integer in the strategy as a minimum
    # score threshold.  Lower values make the strategy more aggressive.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0))
    # External minimum score threshold.  When the absolute score from the
    # indicator composite is below this value the signal is rejected.
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))
    # Trading session filter: start and end times (HH:MM) during which
    # signals may be generated.  Outside these times no new trades are
    # entered.  Note: validation of these fields is performed at runtime.
    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    # ATR multipliers for adaptive stop loss and take profit distances.
    # These values are used by the strategy to scale distances relative
    # to the Average True Range.  They may be tuned via the environment.
    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", 1.5))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", 3.0))

    # Maximum adjustment factors for confidenceâ€‘based scaling of stop loss
    # and take profit.  For example, ``SL_CONFIDENCE_ADJ`` of 0.2 means
    # that a fully confident (score = max) signal will reduce the stop
    # distance by up to 20Â %.  Likewise ``TP_CONFIDENCE_ADJ`` increases
    # the target distance by up to 30Â % for a high confidence signal.
    SL_CONFIDENCE_ADJ: float = float(os.getenv("SL_CONFIDENCE_ADJ", 0.2))
    TP_CONFIDENCE_ADJ: float = float(os.getenv("TP_CONFIDENCE_ADJ", 0.3))

    # Instrument settings
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", 50))
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", 1))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", 5))

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

    # Execution defaults
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # Feature toggles
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ENABLE_TELEGRAM: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")