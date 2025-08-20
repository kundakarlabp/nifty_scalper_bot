# tests/test_position_sizing.py

"""
Tests for the stateless PositionSizer utility.
"""

from __future__ import annotations
import pytest

from src.config import RiskConfig
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession

@pytest.fixture
def risk_config() -> RiskConfig:
    """Provides a default RiskConfig for tests."""
    return RiskConfig(
        risk_per_trade_pct=0.01,  # 1% risk
        max_daily_drawdown_pct=0.05,
        consecutive_loss_limit=3,
        max_trades_per_day=10,
        min_lots=1,
        max_lots=10,
    )

from src.config import ExecutorConfig

@pytest.fixture
def executor_config() -> ExecutorConfig:
    """Provides a default ExecutorConfig for tests."""
    return ExecutorConfig()

@pytest.fixture
def trading_session(risk_config: RiskConfig, executor_config: ExecutorConfig) -> TradingSession:
    """Provides a default TradingSession with 100k equity."""
    return TradingSession(risk_config=risk_config, executor_config=executor_config, starting_equity=100_000.0)


def test_basic_sizing(risk_config: RiskConfig, trading_session: TradingSession):
    """
    Tests basic position sizing calculation.
    - Account: 100k, Risk/Trade: 1% -> Risk Budget: 1000
    - Entry: 200, SL: 180 -> Risk/Contract: 20
    - Max Contracts: 1000 / 20 = 50
    - Lot Size: 50 -> Lots: 1
    - Quantity: 1 * 50 = 50
    """
    sizer = PositionSizer(risk_config)
    quantity = sizer.calculate_quantity(
        session=trading_session,
        entry_price=200.0,
        stop_loss_price=180.0,
        lot_size=50,
    )
    assert quantity == 50

def test_sizing_with_max_lots_clamp(risk_config: RiskConfig, trading_session: TradingSession):
    """
    Tests that the position size is clamped by the max_lots parameter.
    - Account: 100k, Risk/Trade: 1% -> Risk Budget: 1000
    - Entry: 200, SL: 195 -> Risk/Contract: 5
    - Max Contracts: 1000 / 5 = 200
    - Lot Size: 25 -> Lots: 8
    - Max Lots: 5 -> Clamped Lots: 5
    - Quantity: 5 * 25 = 125
    """
    risk_config.max_lots = 5
    risk_config.min_lots = 1
    sizer = PositionSizer(risk_config)
    
    quantity = sizer.calculate_quantity(
        session=trading_session,
        entry_price=200.0,
        stop_loss_price=195.0,
        lot_size=25,
    )
    assert quantity == 125

def test_sizing_with_min_lots_clamp(risk_config: RiskConfig, trading_session: TradingSession):
    """
    Tests that the position size is clamped by the min_lots parameter.
    - Account: 10k, Risk/Trade: 1% -> Risk Budget: 100
    - Entry: 200, SL: 180 -> Risk/Contract: 20
    - Max Contracts: 100 / 20 = 5
    - Lot Size: 25 -> Lots: 0
    - Min Lots: 1 -> Clamped Lots: 1
    - Quantity: 1 * 25 = 25
    """
    trading_session.current_equity = 10_000.0
    risk_config.min_lots = 1
    sizer = PositionSizer(risk_config)
    
    quantity = sizer.calculate_quantity(
        session=trading_session,
        entry_price=200.0,
        stop_loss_price=180.0,
        lot_size=25,
    )
    assert quantity == 25

def test_zero_risk_returns_zero(risk_config: RiskConfig, trading_session: TradingSession):
    """Tests that a zero stop-loss distance results in zero quantity."""
    sizer = PositionSizer(risk_config)
    quantity = sizer.calculate_quantity(
        session=trading_session,
        entry_price=200.0,
        stop_loss_price=200.0,
        lot_size=50,
    )
    assert quantity == 0
