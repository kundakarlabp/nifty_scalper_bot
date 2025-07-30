import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

def calculate_rsi(df: pd.DataFrame, period: int = 14):
    return RSIIndicator(close=df["close"], window=period).rsi()

def calculate_ema(df: pd.DataFrame, period: int):
    return EMAIndicator(close=df["close"], window=period).ema_indicator()

def calculate_macd(df: pd.DataFrame):
    macd_calc = MACD(close=df["close"])
    return {
        "macd": macd_calc.macd(),
        "macd_signal": macd_calc.macd_signal(),
        "macd_hist": macd_calc.macd_diff(),
    }
