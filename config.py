import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Zerodha API Configuration
    ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET", "")
    ZERODHA_CLIENT_ID = os.getenv("ZERODHA_CLIENT_ID", "")
    ZERODHA_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "")
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Market Hours
    MARKET_START_HOUR = int(os.getenv("MARKET_START_HOUR", "9"))
    MARKET_END_HOUR = int(os.getenv("MARKET_END_HOUR", "15"))
    MARKET_START_MINUTE = int(os.getenv("MARKET_START_MINUTE", "15"))
    MARKET_END_MINUTE = int(os.getenv("MARKET_END_MINUTE", "30"))
    
    # Trading Configuration
    AUTO_TRADE = os.getenv("AUTO_TRADE", "true").lower() == "true"
    DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
    REDIS_URL = os.getenv("REDIS_URL", "")
    
    # Signal Configuration
    SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "6.0"))
    MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "5.0"))
    MAX_THRESHOLD = float(os.getenv("MAX_THRESHOLD", "7.0"))
    PERFORMANCE_WINDOW = int(os.getenv("PERFORMANCE_WINDOW", "20"))
    ADAPT_THRESHOLD = os.getenv("ADAPT_THRESHOLD", "true").lower() == "true"
    
    # Capital & Risk Management
    TRADING_CAPITAL = float(os.getenv("TRADING_CAPITAL", "100000"))
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5"))
    TRADE_LOT_SIZE = int(os.getenv("TRADE_LOT_SIZE", "75"))
    MAX_LOSS_STREAK = int(os.getenv("MAX_LOSS_STREAK", "3"))
    LOSS_PAUSE_TIME = int(os.getenv("LOSS_PAUSE_TIME", "60"))
    
    # ATR-based SL/TP
    USE_ATR_SL = os.getenv("USE_ATR_SL", "true").lower() == "true"
    ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.2"))
    USE_ATR_TP = os.getenv("USE_ATR_TP", "true").lower() == "true"
    ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", "1.8"))
    
    # Legacy SL/TP (percentage-based)
    SL_PERCENT = float(os.getenv("SL_PERCENT", "0.20"))
    TP_PERCENT = float(os.getenv("TP_PERCENT", "0.40"))
    
    # News API
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
    
    # Circuit Breaker
    MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "3"))
    LOSS_PAUSE_PCT = float(os.getenv("LOSS_PAUSE_PCT", "3"))
    
    # ML Model
    ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "model.pkl")
    
    # Trading Instrument
    UNDERLYING_SYMBOL = os.getenv("UNDERLYING_SYMBOL", "NIFTY")
    STRIKE_STEP = int(os.getenv("STRIKE_STEP", "50"))
    TRADE_EXCHANGE = os.getenv("TRADE_EXCHANGE", "NFO")
    TRADE_QTY = int(os.getenv("TRADE_QTY", "75"))
    
    # Technical Indicators
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    EMA_FAST = int(os.getenv("EMA_FAST", "9"))
    EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
    MACD_FAST = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
    BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
    BB_STDDEV = float(os.getenv("BB_STDDEV", "2"))
    VOL_SMA_PERIOD = int(os.getenv("VOL_SMA_PERIOD", "20"))
    VWAP_WINDOW = int(os.getenv("VWAP_WINDOW", "0"))
    
    # Flask Configuration
    FLASK_PORT = int(os.getenv("PORT", "10000"))
    FLASK_HOST = os.getenv("HOST", "0.0.0.0")
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        required_fields = [
            'ZERODHA_API_KEY',
            'ZERODHA_API_SECRET', 
            'ZERODHA_CLIENT_ID'
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return True