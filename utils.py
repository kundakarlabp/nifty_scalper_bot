import logging
import os
import pytz
from datetime import datetime, time
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Setup logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def is_market_open(market_start_hour: int = 9, market_start_minute: int = 15,
                  market_end_hour: int = 15, market_end_minute: int = 30) -> bool:
    """Check if market is currently open"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Skip weekends
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    current_time = now.time()
    market_start = time(market_start_hour, market_start_minute)
    market_end = time(market_end_hour, market_end_minute)
    
    return market_start <= current_time <= market_end

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def calculate_position_size(capital: float, risk_percent: float, 
                          entry_price: float, stop_loss: float, 
                          lot_size: int = 75) -> int:
    """Calculate position size based on risk management"""
    if entry_price <= 0 or stop_loss <= 0:
        return lot_size
    
    risk_amount = capital * (risk_percent / 100)
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return lot_size
    
    calculated_qty = int(risk_amount / price_diff)
    
    # Ensure it's in multiples of lot size
    position_lots = max(1, calculated_qty // lot_size)
    return position_lots * lot_size

def format_price(price: float, decimals: int = 2) -> float:
    """Format price to specified decimal places"""
    return round(price, decimals)

class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices.mean() if not prices.empty else 0.0
        
        ema = prices.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        if len(df) < period or 'high' not in df.columns:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            price_mean = prices.mean() if not prices.empty else 0.0
            return {
                'upper': price_mean,
                'middle': price_mean,
                'lower': price_mean
            }
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1]
        }

def validate_trade_data(trade_data: Dict[str, Any]) -> bool:
    """Validate trade data structure"""
    required_fields = ['direction', 'entry_price', 'quantity', 'symbol']
    
    for field in required_fields:
        if field not in trade_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    if trade_data['direction'] not in ['BUY', 'SELL']:
        logger.error(f"Invalid direction: {trade_data['direction']}")
        return False
    
    if safe_float(trade_data['entry_price']) <= 0:
        logger.error(f"Invalid entry price: {trade_data['entry_price']}")
        return False
    
    if safe_int(trade_data['quantity']) <= 0:
        logger.error(f"Invalid quantity: {trade_data['quantity']}")
        return False
    
    return True

class CircuitBreaker:
    """Circuit breaker for trading halts"""
    
    def __init__(self, max_losses: int = 3, pause_minutes: int = 60):
        self.max_losses = max_losses
        self.pause_minutes = pause_minutes
        self.consecutive_losses = 0
        self.is_active = False
        self.resume_time = None
    
    def record_trade(self, pnl: float):
        """Record trade result and update circuit breaker status"""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_losses:
                self.activate()
        else:
            self.consecutive_losses = 0
            self.is_active = False
            self.resume_time = None
    
    def activate(self):
        """Activate circuit breaker"""
        self.is_active = True
        self.resume_time = datetime.now() + pd.Timedelta(minutes=self.pause_minutes)
        logger.warning(f"Circuit breaker activated! Trading paused until {self.resume_time}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.is_active:
            return True
        
        if datetime.now() >= self.resume_time:
            self.is_active = False
            self.consecutive_losses = 0
            self.resume_time = None
            logger.info("Circuit breaker deactivated - trading resumed")
            return True
        
        return False