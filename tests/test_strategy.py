"""
Unit tests for the enhanced scalping strategy.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.signals.signal import Signal



def create_test_dataframe(length: int = 100, trending_up: bool = True, constant_price: bool = False) -> pd.DataFrame:
    """Creates a synthetic OHLCV DataFrame."""
    if constant_price:
        prices = np.full(length, 100.0)
    else:
        prices = np.linspace(100.0, 120.0, length) if trending_up else np.linspace(120.0, 100.0, length)

    data = {
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=length),
    }
    index = pd.date_range(start="2023-01-01", periods=length, freq="min")
    return pd.DataFrame(data, index=pd.to_datetime(index))


def test_generate_signal_returns_valid_structure():
    """A generated signal should be a Signal with valid fields."""
    strategy = EnhancedScalpingStrategy()
    df = create_test_dataframe(trending_up=True)

    sig = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))

    if sig:
        assert isinstance(sig, Signal)
        assert sig.signal in {"BUY", "SELL"}
        assert isinstance(sig.confidence, float)
        assert sig.entry_price > 0
        assert sig.target != sig.stop_loss


def test_no_signal_on_flat_data():
    """No signal when ATR is effectively zero (flat series)."""
    strategy = EnhancedScalpingStrategy()
    df = create_test_dataframe(constant_price=True)

    sig = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert sig is None, "Should not generate a signal when ATR is ~0"


def test_signal_direction_on_trends():
    """
    Up-trend should bias BUY; down-trend should bias SELL,
    subject to scoring/thresholds.
    """
    strategy = EnhancedScalpingStrategy()

    df_up = create_test_dataframe(trending_up=True)
    sig_up = strategy.generate_signal(df_up, current_price=float(df_up['close'].iloc[-1]))
    if sig_up:
        assert sig_up.signal == "BUY"

    df_down = create_test_dataframe(trending_up=False)
    sig_down = strategy.generate_signal(df_down, current_price=float(df_down['close'].iloc[-1]))
    if sig_down:
        assert sig_down.signal == "SELL"
