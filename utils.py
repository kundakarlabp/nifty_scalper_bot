#!/usr/bin/env python3
"""
Utils Module - Utility functions for Nifty Scalper Bot
Contains market timing, data processing, and helper functions
"""

import pytz
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)

# ================================
# MARKET TIMING UTILITIES
# ================================

def is_market_open() -> bool:
    """
    Check if Indian stock market is currently open
    
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE)
        market_end = time(Config.MARKET_END_HOUR, Config.MARKET_END_MINUTE)
        current_time = now.time()
        
        return market_start <= current_time <= market_end
        
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False

def format_price(price):
    """Format price to 2 decimal places"""
    return round(float(price), 2) if price is not None else 0.0

def get_market_status() -> str:
    """
    Get market status emoji and text
    
    Returns:
        str: Market status with emoji
    """
    if is_market_open():
        return "🟢 OPEN"
    else:
        return "🔴 CLOSED"

def time_until_market_open() -> str:
    """
    Get time until market opens
    
    Returns:
        str: Human readable time until market opens
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        if now.weekday() >= 5:  # Weekend
            # Calculate days until Monday
            days_until_monday = 7 - now.weekday()
            if now.weekday() == 5:  # Saturday
                days_until_monday = 2
            elif now.weekday() == 6:  # Sunday
                days_until_monday = 1
            return f"Opens Monday at 9:15 AM (in {days_until_monday} days)"
        
        market_start = now.replace(
            hour=Config.MARKET_START_HOUR, 
            minute=Config.MARKET_START_MINUTE, 
            second=0, 
            microsecond=0
        )
        
        if now.time() < time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE):
            # Market opens today
            diff = market_start - now
            hours, remainder = divmod(diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"Opens in {hours}h {minutes}m"
        else:
            # Market opens tomorrow
            return "Opens tomorrow at 9:15 AM"
            
    except Exception as e:
        logger.error(f"Error calculating time until market open: {e}")
        return "Unknown"

def is_trading_session() -> bool:
    """
    Check if it's during active trading session (excludes first and last 15 minutes)
    
    Returns:
        bool: True if in active trading session
    """
    try:
        if not is_market_open():
            return False
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        current_time = now.time()
        
        # Active trading: 9:30 AM to 3:15 PM
        trading_start = time(9, 30)
        trading_end = time(15, 15)
        
        return trading_start <= current_time <= trading_end
        
    except Exception as e:
        logger.error(f"Error checking trading session: {e}")
        return False

def get_market_session_info() -> Dict[str, Any]:
    """
    Get comprehensive market session information
    
    Returns:
        Dict: Market session details
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        return {
            'is_market_open': is_market_open(),
            'is_trading_session': is_trading_session(),
            'current_time': now.strftime('%H:%M:%S'),
            'market_status': get_market_status(),
            'time_until_open': time_until_market_open() if not is_market_open() else None,
            'is_weekend': now.weekday() >= 5,
            'day_of_week': now.strftime('%A')
        }
        
    except Exception as e:
        logger.error(f"Error getting market session info: {e}")
        return {}

# ================================
# DATA PROCESSING UTILITIES
# ================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for price data
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with technical indicators added
    """
    try:
        if df.empty or 'close' not in df.columns:
            return df
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'])
        
        # MACD
        macd_data = calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of prices
        period: RSI period (default 14)
        
    Returns:
        Series: RSI values
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(index=prices.index)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        Dict: MACD components
    """
    try:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of prices
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Dict: Bollinger band components
    """
    try:
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}

# ================================
# TRADING UTILITIES
# ================================

def calculate_stop_loss(entry_price: float, direction: str, atr: float = None, percentage: float = 0.01) -> float:
    """
    Calculate stop loss price
    
    Args:
        entry_price: Entry price
        direction: BUY or SELL
        atr: Average True Range (optional)
        percentage: Stop loss percentage (default 1%)
        
    Returns:
        float: Stop loss price
    """
    try:
        if atr:
            # ATR-based stop loss
            multiplier = Config.ATR_STOP_MULTIPLIER if hasattr(Config, 'ATR_STOP_MULTIPLIER') else 2.0
            if direction == 'BUY':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        else:
            # Percentage-based stop loss
            if direction == 'BUY':
                return entry_price * (1 - percentage)
            else:
                return entry_price * (1 + percentage)
                
    except Exception as e:
        logger.error(f"Error calculating stop loss: {e}")
        return entry_price * (0.99 if direction == 'BUY' else 1.01)

def calculate_target_price(entry_price: float, stop_loss: float, direction: str, risk_reward_ratio: float = 2.0) -> float:
    """
    Calculate target price based on risk-reward ratio
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: BUY or SELL
        risk_reward_ratio: Risk to reward ratio
        
    Returns:
        float: Target price
    """
    try:
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if direction == 'BUY':
            return entry_price + reward
        else:
            return entry_price - reward
            
    except Exception as e:
        logger.error(f"Error calculating target price: {e}")
        return entry_price

def validate_trade_parameters(entry_price: float, stop_loss: float, target: float, direction: str) -> Tuple[bool, str]:
    """
    Validate trade parameters
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        target: Target price
        direction: Trade direction
        
    Returns:
        Tuple: (is_valid, error_message)
    """
    try:
        if entry_price <= 0:
            return False, "Invalid entry price"
        
        if stop_loss <= 0:
            return False, "Invalid stop loss price"
        
        if target <= 0:
            return False, "Invalid target price"
        
        if direction == 'BUY':
            if stop_loss >= entry_price:
                return False, "Stop loss should be below entry price for BUY"
            if target <= entry_price:
                return False, "Target should be above entry price for BUY"
        elif direction == 'SELL':
            if stop_loss <= entry_price:
                return False, "Stop loss should be above entry price for SELL"
            if target >= entry_price:
                return False, "Target should be below entry price for SELL"
        else:
            return False, "Invalid direction"
        
        return True, "Valid parameters"
        
    except Exception as e:
        logger.error(f"Error validating trade parameters: {e}")
        return False, str(e)

# ================================
# DATA FORMATTING UTILITIES
# ================================

def format_currency(amount: float, currency: str = "₹") -> str:
    """
    Format currency with proper formatting
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    try:
        if abs(amount) >= 10000000:  # 1 crore
            return f"{currency}{amount/10000000:.2f}Cr"
        elif abs(amount) >= 100000:  # 1 lakh
            return f"{currency}{amount/100000:.2f}L"
        elif abs(amount) >= 1000:  # 1 thousand
            return f"{currency}{amount/1000:.2f}K"
        else:
            return f"{currency}{amount:,.2f}"
            
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return f"{currency}{amount:.2f}"

def format_percentage(value: float) -> str:
    """
    Format percentage with proper sign and formatting
    
    Args:
        value: Percentage value
        
    Returns:
        str: Formatted percentage string
    """
    try:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}%"
        
    except Exception as e:
        logger.error(f"Error formatting percentage: {e}")
        return f"{value:.2f}%"

def format_trade_duration(start_time: datetime, end_time: datetime = None) -> str:
    """
    Format trade duration in human readable format
    
    Args:
        start_time: Trade start time
        end_time: Trade end time (default: now)
        
    Returns:
        str: Formatted duration string
    """
    try:
        if end_time is None:
            end_time = datetime.now()
        
        duration = end_time - start_time
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        elif duration.seconds >= 3600:
            return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        else:
            return f"{duration.seconds//60}m {duration.seconds%60}s"
            
    except Exception as e:
        logger.error(f"Error formatting trade duration: {e}")
        return "Unknown"

# ================================
# LOGGING UTILITIES
# ================================

def setup_detailed_logging(log_level: str = "INFO") -> None:
    """
    Setup detailed logging configuration
    
    Args:
        log_level: Logging level
    """
    try:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('detailed_trading.log'),
                logging.StreamHandler()
            ]
        )
        
        # Set specific loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
    except Exception as e:
        print(f"Error setting up logging: {e}")

def log_trade_execution(trade_data: Dict[str, Any], trade_type: str = "entry") -> None:
    """
    Log trade execution details
    
    Args:
        trade_data: Trade data dictionary
        trade_type: Type of trade (entry/exit)
    """
    try:
        logger.info(f"=== TRADE {trade_type.upper()} ===")
        for key, value in trade_data.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 30)
        
    except Exception as e:
        logger.error(f"Error logging trade execution: {e}")

# ================================
# PERFORMANCE UTILITIES
# ================================

def calculate_performance_metrics(trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics
    
    Args:
        trade_history: List of trade records
        
    Returns:
        Dict: Performance metrics
    """
    try:
        if not trade_history:
            return {}
        
        df = pd.DataFrame(trade_history)
        if 'pnl' not in df.columns:
            return {}
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        total_pnl = df['pnl'].sum()
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        # Calculate maximum drawdown
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['running_max'] = df['cumulative_pnl'].expanding().max()
        df['drawdown'] = df['cumulative_pnl'] - df['running_max']
        max_drawdown = df['drawdown'].min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': calculate_sharpe_ratio(df['pnl']) if len(df) > 1 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {}

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default 5%)
        
    Returns:
        float: Sharpe ratio
    """
    try:
        if len(returns) < 2:
            return 0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return excess_returns / returns.std() if returns.std() != 0 else 0
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return 0

# ================================
# HELPER UTILITIES
# ================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        int: Converted value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def get_ist_time() -> datetime:
    """
    Get current IST time
    
    Returns:
        datetime: Current IST time
    """
    try:
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist)
    except Exception as e:
        logger.error(f"Error getting IST time: {e}")
        return datetime.now()

def is_valid_trading_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol to validate
        
    Returns:
        bool: True if valid symbol
    """
    try:
        if not symbol or len(symbol) < 3:
            return False
        
        # Add more validation rules as needed
        return True
        
    except Exception as e:
        logger.error(f"Error validating trading symbol: {e}")
        return False

class TechnicalIndicators:
    @staticmethod
    def sma(prices, period):
        """Simple Moving Average"""
        return sum(prices[-period:]) / period if len(prices) >= period else None
    
    @staticmethod
    def ema(prices, period):
        """Exponential Moving Average"""
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def rsi(prices, period=14):
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
