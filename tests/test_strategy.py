"""
Unit tests for the enhanced scalping strategy.

These tests verify that the strategy returns sensible signal objects on
simple synthetic data.  The goal is not to exhaustively test the
indicator calculations (which are provided by the ``ta`` library) but
to ensure that the wrapper logic behaves correctly.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def create_trending_dataframe(length: int = 100, up: bool = True) -> pd.DataFrame:
    """Create a synthetic OHLC DataFrame trending up or down."""
    import numpy as np
    prices = np.linspace(100, 120, length) if up else np.linspace(120, 100, length)
    data = {
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.ones(length) * 1000,
    }
    return pd.DataFrame(data)


def test_generate_signal_structure() -> None:
    strategy = EnhancedScalpingStrategy(base_stop_loss_points=20, base_target_points=40, confidence_threshold=5)
    df = create_trending_dataframe()
    current_price = df.iloc[-1]["close"]
    signal = strategy.generate_signal(df, current_price)
    assert signal is not None, "Strategy should return a signal on trending data"
    assert signal["signal"] in {"BUY", "SELL"}
    assert 0 <= signal["confidence"] <= 10, "Confidence should be within 0–10"
    assert signal["target"] != signal["stop_loss"], "Target and stop loss should differ"


def test_signal_direction_changes() -> None:
    strategy = EnhancedScalpingStrategy(base_stop_loss_points=20, base_target_points=40, confidence_threshold=5)
    df_up = create_trending_dataframe(up=True)
    df_down = create_trending_dataframe(up=False)
    price_up = df_up.iloc[-1]["close"]
    price_down = df_down.iloc[-1]["close"]
    signal_up = strategy.generate_signal(df_up, price_up)
    signal_down = strategy.generate_signal(df_down, price_down)
    assert signal_up["signal"] != signal_down["signal"], "Opposite trends should yield opposite signals"