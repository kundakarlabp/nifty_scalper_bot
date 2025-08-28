"""Unit tests for the enhanced scalping strategy."""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

from src.config import StrategySettings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.indicators import calculate_adx


@pytest.fixture
def strategy_config() -> StrategySettings:
    """Provides a default StrategySettings for tests."""
    return StrategySettings(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )


def create_test_dataframe(length: int = 100, trending_up: bool = True, constant_price: bool = False) -> pd.DataFrame:
    """Creates a synthetic OHLCV DataFrame."""
    if constant_price:
        prices = np.full(length, 100.0)
        data = {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.random.randint(100, 1000, size=length),
        }
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
    df = pd.DataFrame(data, index=pd.to_datetime(index))
    adx, di_plus, di_minus = calculate_adx(df)
    df["adx"] = adx
    df["di_plus"] = di_plus
    df["di_minus"] = di_minus
    return df


def create_low_vol_dataframe(length: int = 100) -> pd.DataFrame:
    """Create a narrow-range DataFrame to simulate indecisive markets."""
    base = 100.0
    prices = base + np.sin(np.linspace(0, 2 * np.pi, length)) * 0.1
    data = {
        "open": prices,
        "high": prices + 0.1,
        "low": prices - 0.1,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=length),
    }
    index = pd.date_range(start="2023-01-01", periods=length, freq="min")
    return pd.DataFrame(data, index=pd.to_datetime(index))


def test_generate_signal_returns_valid_structure(strategy_config: StrategySettings):
    """A generated signal should be a dict with valid fields."""
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )
    df = create_test_dataframe(trending_up=True)

    sig = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))

    if sig:
        assert isinstance(sig, dict)
        assert sig["side"] in {"BUY", "SELL"}
        assert isinstance(sig["confidence"], float)
        assert sig["entry_price"] > 0
        assert sig["target"] != sig["stop_loss"]


def test_no_signal_on_flat_data(strategy_config: StrategySettings):
    """No signal when ATR is effectively zero (flat series)."""
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )
    df = create_test_dataframe(constant_price=True)

    sig = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert sig is None, "Should not generate a signal when ATR is ~0"


def test_signal_direction_on_trends(strategy_config: StrategySettings):
    """
    Up-trend should bias BUY; down-trend should bias SELL,
    subject to scoring/thresholds.
    """
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )

    df_up = create_test_dataframe(trending_up=True)
    sig_up = strategy.generate_signal(df_up, current_price=float(df_up['close'].iloc[-1]))
    # Ensure signal generated
    assert sig_up is not None
    assert sig_up["side"] == "BUY"

    df_down = create_test_dataframe(trending_up=False)
    sig_down = strategy.generate_signal(df_down, current_price=float(df_down['close'].iloc[-1]))
    # Ensure signal generated
    assert sig_down is not None
    assert sig_down["side"] == "SELL"


def test_no_trade_when_indecisive(strategy_config: StrategySettings):
    """Strategy should abstain when ADX and BB width show indecision."""
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )
    df = create_low_vol_dataframe()
    sig = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert sig is None
