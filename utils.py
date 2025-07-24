#!/usr/bin/env python3
"""
Utility functions and helpers for Nifty Scalper Bot
"""

import logging
import os
import pytz
from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional
import pandas as pd
from config import Config

# --- Logging setup ---
def setup_logging() -> logging.Logger:
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Market timing utilities ---
def is_market_open() -> bool:
    """
    Check if market is currently open (9:15–15:30 IST, weekdays).
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        # Skip weekends
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_start = time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE)
        market_end = time(Config.MARKET_END_HOUR, Config.MARKET_END_MINUTE)
        return market_start <= now.time() <= market_end
    except Exception as e:
        logger.error(f"is_market_open error: {e}")
        return False

def get_market_status() -> str:
    """Return market status as emoji + text."""
    return "🟢 OPEN" if is_market_open() else "🔴 CLOSED"

def time_until_market_open() -> str:
    """
    Return a human-friendly message indicating time until next market open.
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        # Weekend handling
        if now.weekday() >= 5:
            days = 7 - now.weekday()
            return f"Opens Monday at 9:15 AM (in {days} days)"
        # Today
        today_open = now.replace(
            hour=Config.MARKET_START_HOUR,
            minute=Config.MARKET_START_MINUTE,
            second=0,
            microsecond=0
        )
        if now.time() < today_open.time():
            diff = today_open - now
            hrs, rem = divmod(diff.seconds, 3600)
            mins, _ = divmod(rem, 60)
            return f"Opens in {hrs}h {mins}m"
        # Tomorrow
        return "Opens tomorrow at 9:15 AM"
    except Exception as e:
        logger.error(f"time_until_market_open error: {e}")
        return "Unknown"

# --- Safe converters ---
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, with fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# --- Position sizing ---
def calculate_position_size(
    capital: float,
    risk_fraction: float,
    entry_price: float,
    stop_loss: float,
    lot_size: int = Config.DEFAULT_LOT_SIZE
) -> int:
    """
    Calculate position size in lots based on risk_fraction (e.g., 0.01 for 1%).
    Ensures at least one lot, and rounds down to nearest lot.
    """
    if entry_price <= 0 or stop_loss <= 0:
        return lot_size
    risk_amount = capital * risk_fraction
    price_diff = abs(entry_price - stop_loss)
    if price_diff <= 0:
        return lot_size
    raw_qty = int(risk_amount / price_diff)
    lots = max(1, raw_qty // lot_size)
    return lots * lot_size

def format_price(price: float, decimals: int = 2) -> float:
    """Round price to given number of decimal places."""
    return round(price, decimals)

# --- Technical indicators ---
class TechnicalIndicators:
    """Calculations for common technical indicators."""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1] if not rsi.empty else 50.0)

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> float:
        if len(prices) < period:
            return float(prices.mean() if not prices.empty else 0.0)
        ema = prices.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
        }

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period or not all(col in df.columns for col in ['high', 'low', 'close']):
            return 0.0
        high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1] if not atr.empty else 0.0)

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, float]:
        if len(prices) < period:
            mean = float(prices.mean() if not prices.empty else 0.0)
            return {'upper': mean, 'middle': mean, 'lower': mean}
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return {
            'upper': float((sma + std * std_dev).iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float((sma - std * std_dev).iloc[-1]),
        }

# --- Trade data validation ---
def validate_trade_data(trade_data: Dict[str, Any]) -> bool:
    """
    Ensure a trade_data dict has required fields and valid values.
    """
    required = ['direction', 'entry_price', 'quantity', 'symbol']
    for field in required:
        if field not in trade_data:
            logger.error(f"Missing required field: {field}")
            return False
    if trade_data['direction'] not in ('BUY', 'SELL'):
        logger.error(f"Invalid direction: {trade_data['direction']}")
        return False
    if safe_float(trade_data['entry_price']) <= 0 or safe_int(trade_data['quantity']) <= 0:
        logger.error(f"Invalid trade values: {trade_data}")
        return False
    return True

# --- Circuit breaker ---
class CircuitBreaker:
    """
    Pauses trading after a sequence of losing trades.
    """
    def __init__(self, max_losses: int = Config.CIRCUIT_BREAKER_MAX_LOSSES,
                       pause_minutes: int = Config.CIRCUIT_BREAKER_PAUSE_MINUTES):
        self.max_losses = max_losses
        self.pause_minutes = pause_minutes
        self.consecutive_losses = 0
        self.is_active = False
        self.resume_time: Optional[datetime] = None

    def record_trade(self, pnl: float):
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_losses:
                self.activate()
        else:
            self.consecutive_losses = 0

    def activate(self):
        self.is_active = True
        self.resume_time = datetime.now() + timedelta(minutes=self.pause_minutes)
        logger.warning(f"Circuit breaker activated until {self.resume_time}")

    def can_trade(self) -> bool:
        if not self.is_active:
            return True
        if datetime.now() >= self.resume_time:
            self.is_active = False
            self.consecutive_losses = 0
            self.resume_time = None
            logger.info("Circuit breaker cleared")
            return True
        return False
