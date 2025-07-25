import pandas as pd
import pandas_ta as ta
from datetime import datetime, time, timedelta
import pytz
from typing import Dict, Any

# Define the Indian Standard Time (IST) timezone for market timings
IST = pytz.timezone('Asia/Kolkata')


# --- 1. Technical Indicators Class ---
# This class centralizes all indicator calculations using the pandas-ta library.
class TechnicalIndicators:
    """
    A utility class to calculate various technical indicators using pandas-ta.
    This ensures all indicator logic is centralized, tested, and correct.
    """

    def calculate_rsi(self, close_prices: pd.Series, period: int) -> float:
        """Calculates the Relative Strength Index (RSI)."""
        if close_prices.empty or len(close_prices) < period:
            return 50.0  # Return a neutral value if not enough data
        rsi = ta.rsi(close=close_prices, length=period)
        return rsi.iloc[-1] if not rsi.empty else 50.0

    def calculate_ema(self, close_prices: pd.Series, period: int) -> float:
        """Calculates the Exponential Moving Average (EMA)."""
        if close_prices.empty or len(close_prices) < period:
            return close_prices.iloc[-1] if not close_prices.empty else 0.0
        ema = ta.ema(close=close_prices, length=period)
        return ema.iloc[-1] if not ema.empty else 0.0

    def calculate_macd(self, close_prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, float]:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if close_prices.empty or len(close_prices) < slow:
            return {'macd': 0.0, 'histogram': 0.0, 'signal': 0.0}
            
        # pandas-ta returns a DataFrame with columns like 'MACD_12_26_9'
        macd_df = ta.macd(close=close_prices, fast=fast, slow=slow, signal=signal)
        if macd_df is None or macd_df.empty:
            return {'macd': 0.0, 'histogram': 0.0, 'signal': 0.0}
            
        # Extract the last row of values safely
        last_values = macd_df.iloc[-1]
        return {
            'macd': last_values.get(f'MACD_{fast}_{slow}_{signal}', 0.0),
            'histogram': last_values.get(f'MACDh_{fast}_{slow}_{signal}', 0.0),
            'signal': last_values.get(f'MACDs_{fast}_{slow}_{signal}', 0.0)
        }

    def calculate_bollinger_bands(self, close_prices: pd.Series, period: int, stddev: float) -> Dict[str, float]:
        """Calculates Bollinger Bands (Upper, Middle, Lower)."""
        if close_prices.empty or len(close_prices) < period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
            
        bbands_df = ta.bbands(close=close_prices, length=period, std=stddev)
        if bbands_df is None or bbands_df.empty:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
            
        last_values = bbands_df.iloc[-1]
        return {
            'upper': last_values.get(f'BBU_{period}_{stddev:.1f}', 0.0),
            'middle': last_values.get(f'BBM_{period}_{stddev:.1f}', 0.0),
            'lower': last_values.get(f'BBL_{period}_{stddev:.1f}', 0.0)
        }

    def calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculates the Average True Range (ATR). Requires 'high', 'low', 'close' columns."""
        if df.empty or len(df) < period or not all(k in df for k in ['high', 'low', 'close']):
            return 0.0
        atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=period)
        return atr.iloc[-1] if atr is not None and not atr.empty else 0.0


# --- 2. Market Timing and Status Functions ---

def get_market_session_info() -> Dict[str, str]:
    """Gets information about the current market session."""
    now_ist = datetime.now(IST)
    return {
        'current_time': now_ist.strftime('%H:%M:%S'),
        'day_of_week': now_ist.strftime('%A')
    }

def is_market_open(market_start_time=time(9, 15), market_end_time=time(15, 30)) -> bool:
    """Checks if the Indian stock market is currently open."""
    now_ist = datetime.now(IST)
    # Market is open on weekdays (Monday=0, Sunday=6)
    if now_ist.weekday() >= 5:
        return False
    # Market is open between specified hours
    return market_start_time <= now_ist.time() <= market_end_time

def get_market_status() -> str:
    """Returns a user-friendly string for the market status."""
    return "🟢 OPEN" if is_market_open() else "🔴 CLOSED"

def time_until_market_open(market_start_time=time(9, 15)) -> timedelta:
    """Calculates the time remaining until the market opens."""
    now_ist = datetime.now(IST)
    if is_market_open():
        return timedelta(0)
    
    # Calculate next opening time
    open_time = datetime.combine(now_ist.date(), market_start_time, tzinfo=IST)
    if now_ist.time() > market_start_time or now_ist.weekday() >= 5:
        # If it's after market hours or weekend, find the next weekday
        days_to_add = 1
        if now_ist.weekday() == 4: days_to_add = 3 # Friday -> Monday
        elif now_ist.weekday() == 5: days_to_add = 2 # Saturday -> Monday
        open_time = datetime.combine(now_ist.date() + timedelta(days=days_to_add), market_start_time, tzinfo=IST)
        
    return open_time - now_ist


# --- 3. Formatting and Calculation Helpers ---

def format_currency(amount: float) -> str:
    """Formats a float into a currency string (e.g., ₹1,00,000.50)."""
    return f"₹{amount:,.2f}"

def format_percentage(value: float) -> str:
    """Formats a float into a percentage string (e.g., 5.25%)."""
    return f"{value:.2f}%"

def format_trade_duration(duration: timedelta) -> str:
    """Formats a timedelta object into a readable string (e.g., 1h 15m 30s)."""
    seconds = int(duration.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def calculate_performance_metrics(current_balance, initial_balance, todays_pnl, trades=[]) -> Dict[str, Any]:
    """A placeholder for calculating performance metrics like win rate and profit factor."""
    # This is a simplified version. A real implementation would need a history of trades.
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
    total_trades = len(trades)
    
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    total_loss = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
    
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
