"""
Tests for the TradingSession state manager.
"""

import pytest
from src.config import RiskConfig
from src.risk.session import TradingSession, Trade

@pytest.fixture
def risk_config() -> RiskConfig:
    """Provides a default RiskConfig for tests."""
    return RiskConfig(
        max_daily_drawdown_pct=0.05,  # 5%
        consecutive_loss_limit=3,
        max_trades_per_day=5,
        risk_per_trade_pct=0.01,
        min_lots=1,
        max_lots=10,
    )

from src.config import ExecutorConfig

@pytest.fixture
def executor_config() -> ExecutorConfig:
    """Provides a default ExecutorConfig for tests."""
    return ExecutorConfig()

@pytest.fixture
def session(risk_config: RiskConfig, executor_config: ExecutorConfig) -> TradingSession:
    """Provides a TradingSession with 100k equity."""
    return TradingSession(risk_config=risk_config, executor_config=executor_config, starting_equity=100_000.0)


def test_session_initialization(session: TradingSession):
    assert session.start_equity == 100_000.0
    assert session.current_equity == 100_000.0
    assert session.daily_pnl == 0.0
    assert session.consecutive_losses == 0
    assert session.trades_today == 0
    assert not session.active_trades
    assert not session.trade_history

def test_add_trade(session: TradingSession):
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr=10.0)
    session.add_trade(trade)

    assert session.trades_today == 1
    assert "order1" in session.active_trades
    assert session.active_trades["order1"] == trade

def test_finalize_winning_trade(session: TradingSession):
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr=10.0)
    session.add_trade(trade)
    session.finalize_trade(order_id="order1", exit_price=220.0)

    assert not session.active_trades
    assert len(session.trade_history) == 1
    assert session.daily_pnl == (220.0 - 200.0) * 50
    assert session.current_equity == 100_000.0 + session.daily_pnl
    assert session.consecutive_losses == 0

def test_finalize_losing_trade_and_consecutive_losses(session: TradingSession):
    # First loss
    trade1 = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr=10.0)
    session.add_trade(trade1)
    session.finalize_trade(order_id="order1", exit_price=190.0)
    assert session.consecutive_losses == 1
    assert session.daily_pnl == -500.0

    # Second loss
    trade2 = Trade("NIFTY_PE", "BUY", 150.0, 50, "order2", atr=8.0)
    session.add_trade(trade2)
    session.finalize_trade(order_id="order2", exit_price=145.0)
    assert session.consecutive_losses == 2
    assert session.daily_pnl == -500.0 + (-250.0)

    # A winning trade should reset the counter
    trade3 = Trade("NIFTY_CE", "BUY", 200.0, 50, "order3", atr=10.0)
    session.add_trade(trade3)
    session.finalize_trade(order_id="order3", exit_price=210.0)
    assert session.consecutive_losses == 0

def test_risk_limit_max_trades(session: TradingSession):
    for i in range(5):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 201.0)

    assert session.trades_today == 5
    assert session.check_risk_limits() is None # Limit is not reached yet

    # The 6th trade should be blocked
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order6", atr=10.0)
    session.add_trade(trade)
    assert session.check_risk_limits() is not None

def test_risk_limit_consecutive_loss(session: TradingSession):
    for i in range(3):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 199.0) # 3 losses

    assert session.consecutive_losses == 3
    assert session.check_risk_limits() is not None # Limit reached

def test_risk_limit_max_drawdown(session: TradingSession):
    # 5% of 100k is 5k
    # Each trade loses (200 - 180) * 50 = 1000
    for i in range(5):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 180.0)

    assert session.daily_pnl == -5000.0
    assert session.drawdown_pct == 0.05
    assert session.check_risk_limits() is not None # Limit reached
