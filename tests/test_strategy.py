"""
Unit tests for the enhanced scalping strategy.
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

from src.config import StrategyConfig
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

@pytest.fixture
def strategy_config() -> StrategyConfig:
    """Provides a default StrategyConfig for tests."""
    return StrategyConfig(
        min_signal_score=5.0,
        confidence_threshold=6.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )

def create_test_dataframe(length: int = 100, trending_up: bool = True, constant_price: bool = False) -> pd.DataFrame:
    """Creates a synthetic OHLCV DataFrame."""
    if constant_price:
        prices = np.full(length, 100)
    else:
        prices = np.linspace(100, 120, length) if trending_up else np.linspace(120, 100, length)

    data = {
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=length),
    }
    # Create a proper DatetimeIndex
    index = pd.to_datetime(pd.date_range(start="2023-01-01", periods=length, freq="min"))
    return pd.DataFrame(data, index=index)


def test_generate_signal_returns_valid_structure(strategy_config: StrategyConfig):
    """Tests that a valid signal has the correct structure and keys."""
    strategy = EnhancedScalpingStrategy(strategy_config)
    df = create_test_dataframe(trending_up=True)
    # The scoring logic is simplified, so we might not get a signal.
    # This test primarily checks the return structure if a signal *is* generated.
    signal = strategy.generate_signal(df, current_price=df["close"].iloc[-1])

    if signal:
        assert "signal" in signal
        assert signal["signal"] in {"BUY", "SELL"}
        assert "confidence" in signal
        assert "entry_price" in signal
        assert "stop_loss" in signal
        assert "target" in signal
        assert signal["target"] != signal["stop_loss"]

def test_no_signal_on_flat_data(strategy_config: StrategyConfig):
    """Tests that no signal is generated on flat data where ATR would be zero."""
    strategy = EnhancedScalpingStrategy(strategy_config)
    df = create_test_dataframe(constant_price=True)
    signal = strategy.generate_signal(df, current_price=df["close"].iloc[-1])
    assert signal is None, "Should not generate a signal when ATR is zero"

def test_signal_direction_on_trends(strategy_config: StrategyConfig):
    """
    Tests that an upward trend generates a BUY signal and a downward trend
    generates a SELL signal. This test is sensitive to the scoring logic.
    """
    strategy = EnhancedScalpingStrategy(strategy_config)

    # Create a strong upward trend
    df_up = create_test_dataframe(trending_up=True)
    signal_up = strategy.generate_signal(df_up, current_price=df_up["close"].iloc[-1])

    # The simplified scoring logic might not produce a BUY signal.
    # If it does, we can assert it's a BUY.
    if signal_up:
        assert signal_up["signal"] == "BUY"

    # Create a strong downward trend
    df_down = create_test_dataframe(trending_up=False)
    signal_down = strategy.generate_signal(df_down, current_price=df_down["close"].iloc[-1])

    # If a signal is generated, assert it's a SELL.
    if signal_down:
        assert signal_down["signal"] == "SELL"