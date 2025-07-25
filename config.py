# config.py

import os

class Config:
    """
    A centralized and consolidated configuration class for the Nifty Scalper Bot.
    It sources settings from environment variables for security and flexibility.
    """
    
    # --- SECTION 1: CRITICAL CREDENTIALS & IDENTIFIERS ---
    # These MUST be set in your environment.
    ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY')
    ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET')
    ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN') # For session management
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_ADMIN_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') # Your personal chat ID for critical alerts

    # --- SECTION 2: CORE TRADING & RISK PARAMETERS ---
    # These define the bot's primary financial behavior.
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000.0))
    RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', 0.01)) # Risk 1% of account per trade.

    # --- SECTION 3: SYSTEM-WIDE SAFETY LIMITS (CAPITAL PRESERVATION) ---
    # These are the bot's main circuit breakers.
    MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', 0.03)) # Stop trading if down 3% on the day.
    MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', 10))         # Max trades per day.
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3)) # Trigger timeout after 3 straight losses.
    CIRCUIT_BREAKER_PAUSE_MINUTES = int(os.getenv('CIRCUIT_BREAKER_PAUSE_MINUTES', 30))

    # --- SECTION 4: TRADE EXECUTION & INSTRUMENT DETAILS ---
    # Parameters related to what and how the bot trades.
    NIFTY_LOT_SIZE = int(os.getenv('NIFTY_LOT_SIZE', 25)) # Current Nifty F&O lot size.
    
    # The new SignalGenerator uses ATR for dynamic SL/TP. These are the multipliers.
    # This setup creates a default Risk-Reward Ratio of 1:2.
    ATR_SL_MULT = float(os.getenv('ATR_SL_MULT', 1.5)) # Stop-loss will be 1.5 * ATR.
    ATR_TP_MULT = float(os.getenv('ATR_TP_MULT', 3.0)) # Target will be 3.0 * ATR.

    # --- SECTION 5: TECHNICAL INDICATOR SETTINGS ---
    # Tune the "brains" of the bot here.
    EMA_FAST = int(os.getenv('EMA_FAST', 10))
    EMA_SLOW = int(os.getenv('EMA_SLOW', 30))
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', 14))
    ATR_PERIOD = int(os.getenv('ATR_PERIOD', 14))
    # Note: MACD and BBands are not used in the new, simplified high-conviction signal generator,
    # but are kept here in case you want to re-integrate them.
    MACD_FAST = int(os.getenv('MACD_FAST', 12))
    MACD_SLOW = int(os.getenv('MACD_SLOW', 26))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', 9))
    BB_PERIOD = int(os.getenv('BB_PERIOD', 20))
    BB_STDDEV = float(os.getenv('BB_STDDEV', 2.0))

    # --- SECTION 6: BOT OPERATION & ENVIRONMENT ---
    # General operational settings.
    TICK_INTERVAL_SECONDS = int(os.getenv('TICK_INTERVAL_SECONDS', 3)) # How often the main loop runs.
    STATE_FILE = os.getenv('STATE_FILE', "position_state.json") # File for saving live trade state.
    DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true' # If True, will not place real orders.

    # --- SECTION 7: ADVANCED & EXTERNAL SERVICES (Optional) ---
    # Configuration for Redis, Webhooks, etc. The bot logic will need to be
    # extended to use these.
    REDIS_URL = os.getenv('REDIS_URL') # e.g., 'redis://localhost:6379'
    WEBHOOK_URL = os.getenv('WEBHOOK_URL') # For broker postbacks

    # --- VALIDATION METHOD ---
    @classmethod
    def validate(cls):
        """Validates that critical environment variables are set."""
        required_vars = [
            'ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_ADMIN_CHAT_ID'
        ]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"FATAL: Missing required environment variables: {', '.join(missing)}")
        
        if cls.RISK_PER_TRADE_PCT > 0.05: # Risking more than 5% is dangerous
            print("WARNING: RISK_PER_TRADE_PCT is set to >5%, which is highly aggressive.")
        
        logger.info("Configuration loaded and validated successfully.")
        return True

# --- Optional: A logger instance for use within the config file itself if needed ---
import logging
logger = logging.getLogger(__name__)
