"""Bracket target coverage for elite strategies."""

from __future__ import annotations

from datetime import datetime

from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import (
    BBSqueezeStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import (
    GammaScalpingStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.oi_max_pain import (
    OIMaxPainStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import (
    OrderFlowStrategy,
)


def _assert_brackets(signal, entry: float) -> None:
    """Args: signal, entry. Returns: None. Raises: AssertionError."""
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    assert signal.metadata.get("tp1") is not None
    assert signal.metadata.get("tp2") is not None
    assert signal.take_profit == signal.metadata["tp2"]
    if signal.action == "BUY":
        assert signal.stop_loss < entry
        assert signal.metadata["tp1"] > entry
        assert signal.metadata["tp2"] > signal.metadata["tp1"]
    else:
        assert signal.stop_loss > entry
        assert signal.metadata["tp1"] < entry
        assert signal.metadata["tp2"] < signal.metadata["tp1"]


def test_gamma_scalping_brackets() -> None:
    """Args: none. Returns: None. Raises: AssertionError."""
    config = GammaScalpingStrategyConfig(enabled=True, min_gamma=0.0005, quantity=1)
    strategy = GammaScalpingStrategy(config=config, indicator_engine=None)
    entry = 100.0
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "gamma": 0.003,
            "theta": -5.0,
            "delta": 0.5,
            "macd": 1.2,
            "macd_signal": 0.2,
            "atr": 2.0,
            "volume": 2000.0,
            "avg_volume": 1000.0,
        },
        current_price=entry,
        position=None,
    )
    assert signal is not None
    _assert_brackets(signal, entry)


def test_orb_pro_brackets() -> None:
    """Args: none. Returns: None. Raises: AssertionError."""
    config = ORBProStrategyConfig(enabled=True, orb_minutes=15, quantity=1)
    strategy = ORBProStrategy(config=config, indicator_engine=None)
    formation_ts = datetime(2024, 1, 2, 9, 20)
    breakout_ts = datetime(2024, 1, 2, 9, 40)
    formation = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "high": 100.0,
            "low": 95.0,
            "volume": 1200.0,
            "avg_volume": 1000.0,
            "vwap": 98.0,
            "atr": 2.0,
            "timestamp": formation_ts,
        },
        current_price=97.0,
        position=None,
    )
    assert formation is None
    entry = 110.0
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "high": 110.0,
            "low": 104.0,
            "volume": 2000.0,
            "avg_volume": 1000.0,
            "vwap": 105.0,
            "atr": 2.0,
            "timestamp": breakout_ts,
        },
        current_price=entry,
        position=None,
    )
    assert signal is not None
    _assert_brackets(signal, entry)


def test_order_flow_brackets() -> None:
    """Args: none. Returns: None. Raises: AssertionError."""
    config = OrderFlowStrategyConfig(enabled=True, quantity=1)
    strategy = OrderFlowStrategy(config=config, indicator_engine=None)
    entry = 102.0
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "depth": {
                "buy": [{"price": 101.0, "quantity": 600.0}],
                "sell": [{"price": 103.0, "quantity": 150.0}],
            },
            "vwap": 100.0,
            "atr": 1.5,
            "volume": 1500.0,
            "avg_volume": 1000.0,
        },
        current_price=entry,
        position=None,
    )
    assert signal is not None
    _assert_brackets(signal, entry)


def test_bb_squeeze_brackets() -> None:
    """Args: none. Returns: None. Raises: AssertionError."""
    config = BBSqueezeStrategyConfig(enabled=True, quantity=1)
    strategy = BBSqueezeStrategy(config=config, indicator_engine=None)
    entry = 106.0
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "bollinger_upper": 105.0,
            "bollinger_lower": 95.0,
            "bollinger_middle": 100.0,
            "atr": 2.0,
            "volume": 2000.0,
            "avg_volume": 1000.0,
        },
        current_price=entry,
        position=None,
    )
    assert signal is not None
    _assert_brackets(signal, entry)


def test_oi_max_pain_brackets() -> None:
    """Args: none. Returns: None. Raises: AssertionError."""
    config = OIMaxPainStrategyConfig(enabled=True, quantity=1)
    strategy = OIMaxPainStrategy(config=config, indicator_engine=None)
    entry = 95.0
    signal = strategy.generate_signal(
        symbol="NIFTY",
        indicators={
            "max_pain": 105.0,
            "vwap": 100.0,
            "macd_hist": 0.8,
            "atr": 1.5,
            "timestamp": datetime(2024, 1, 2, 14, 0),
        },
        current_price=entry,
        position=None,
    )
    assert signal is not None
    _assert_brackets(signal, entry)
