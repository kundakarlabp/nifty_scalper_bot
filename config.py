import os

class Config:
    # ===== ZERODHA API CONFIGURATION =====
    ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY')
    ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET')
    ZERODHA_CLIENT_ID = os.getenv('ZERODHA_CLIENT_ID')
    ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    # ===== TELEGRAM BOT CONFIGURATION =====
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # ===== MARKET HOURS =====
    MARKET_START_HOUR = int(os.getenv('MARKET_START_HOUR', 9))
    MARKET_END_HOUR = int(os.getenv('MARKET_END_HOUR', 15))
    MARKET_START_MINUTE = int(os.getenv('MARKET_START_MINUTE', 15))
    MARKET_END_MINUTE = int(os.getenv('MARKET_END_MINUTE', 30))
    
    # ===== TRADING CONFIGURATION =====
    AUTO_TRADE = os.getenv('AUTO_TRADE', 'true').lower() == 'true'
    DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
    WEBHOOK_URL = os.getenv('WEBHOOK_URL')
    
    # ===== REDIS CONFIGURATION =====
    REDIS_URL = os.getenv('REDIS_URL')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
    
    # ===== SIGNAL CONFIGURATION =====
    SIGNAL_THRESHOLD = float(os.getenv('SIGNAL_THRESHOLD', 6.0))
    MIN_THRESHOLD = float(os.getenv('MIN_THRESHOLD', 5.0))
    MAX_THRESHOLD = float(os.getenv('MAX_THRESHOLD', 7.0))
    PERFORMANCE_WINDOW = int(os.getenv('PERFORMANCE_WINDOW', 20))
    ADAPT_THRESHOLD = os.getenv('ADAPT_THRESHOLD', 'true').lower() == 'true'
    
    # ===== CAPITAL & RISK MANAGEMENT =====
    TRADING_CAPITAL = float(os.getenv('TRADING_CAPITAL', 100000))
    MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', 5)) / 100  # Convert to decimal
    TRADE_LOT_SIZE = int(os.getenv('TRADE_LOT_SIZE', 75))
    MAX_LOSS_STREAK = int(os.getenv('MAX_LOSS_STREAK', 3))
    LOSS_PAUSE_TIME = int(os.getenv('LOSS_PAUSE_TIME', 60))  # minutes
    
    # ===== POSITION SIZING (LOTS) =====
    DEFAULT_LOTS = 1
    MIN_LOTS = 1
    MAX_LOTS = int(TRADING_CAPITAL / 50000)  # Max lots based on capital (adjust multiplier as needed)
    LOT_SIZE = TRADE_LOT_SIZE
    
    # ===== ATR-BASED SL/TP =====
    USE_ATR_SL = os.getenv('USE_ATR_SL', 'true').lower() == 'true'
    ATR_SL_MULT = float(os.getenv('ATR_SL_MULT', 1.2))
    USE_ATR_TP = os.getenv('USE_ATR_TP', 'true').lower() == 'true'
    ATR_TP_MULT = float(os.getenv('ATR_TP_MULT', 1.8))
    
    # ===== LEGACY PERCENTAGE-BASED SL/TP =====
    SL_PERCENT = float(os.getenv('SL_PERCENT', 0.20)) / 100  # Convert to decimal
    TP_PERCENT = float(os.getenv('TP_PERCENT', 0.40)) / 100  # Convert to decimal
    
    # ===== NEWS API =====
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    
    # ===== CIRCUIT BREAKER =====
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSEC_LOSSES', 3))
    CIRCUIT_BREAKER_PAUSE_MINUTES = int(os.getenv('LOSS_PAUSE_TIME', 60))
    
    # ===== ML MODEL =====
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'model.pkl')
    
    # ===== TRADING INSTRUMENT =====
    UNDERLYING_SYMBOL = os.getenv('UNDERLYING_SYMBOL', 'NIFTY')
    STRIKE_STEP = int(os.getenv('STRIKE_STEP', 50))
    TRADE_EXCHANGE = os.getenv('TRADE_EXCHANGE', 'NFO')
    TRADE_QTY = int(os.getenv('TRADE_QTY', 75))  # Quantity per lot
    
    # ===== TECHNICAL INDICATORS =====
    ATR_PERIOD = int(os.getenv('ATR_PERIOD', 14))
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', 14))
    EMA_FAST = int(os.getenv('EMA_FAST', 9))
    EMA_SLOW = int(os.getenv('EMA_SLOW', 21))
    MACD_FAST = int(os.getenv('MACD_FAST', 12))
    MACD_SLOW = int(os.getenv('MACD_SLOW', 26))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', 9))
    BB_PERIOD = int(os.getenv('BB_PERIOD', 20))
    BB_STDDEV = float(os.getenv('BB_STDDEV', 2))
    VOL_SMA_PERIOD = int(os.getenv('VOL_SMA_PERIOD', 20))
    VWAP_WINDOW = int(os.getenv('VWAP_WINDOW', 0))
    
    # ===== FLASK CONFIGURATION =====
    PORT = int(os.getenv('PORT', 10000))
    HOST = os.getenv('HOST', '0.0.0.0')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # ===== DERIVED CONFIGURATIONS =====
    RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
    MIN_SIGNAL_INTERVAL = 300  # 5 minutes
    LOOP_DELAY = 5  # seconds
    DEFAULT_QUANTITY = TRADE_QTY  # For backward compatibility
    MIN_QUANTITY = TRADE_QTY
    MAX_QUANTITY = MAX_LOTS * TRADE_QTY
    
    # ===== VALIDATION =====
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        required_vars = [
            'ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 
            'ZERODHA_CLIENT_ID', 'TELEGRAM_BOT_TOKEN'
        ]
        
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        if cls.MAX_DAILY_LOSS_PCT > 0.1:  # More than 10%
            raise ValueError("MAX_DAILY_LOSS_PCT seems too high (>10%)")
        
        if cls.TRADING_CAPITAL < 10000:
            raise ValueError("TRADING_CAPITAL seems too low (<10,000)")
        
        return True
