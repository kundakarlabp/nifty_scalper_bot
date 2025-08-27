"""
Tests for the TradingSession state manager.
"""

import pytest
from src.config import ExecutorSettings, RiskSettings
from src.risk.session import TradingSession, Trade


@pytest.fixture
def risk_config() -> RiskSettings:
    """Provides a default RiskSettings for tests."""
    return RiskSettings(
        max_daily_drawdown_pct=0.05,  # 5%
        consecutive_loss_limit=3,
        max_trades_per_day=5,
        risk_per_trade=0.01,
    )


@pytest.fixture
def executor_config() -> ExecutorSettings:
    """Provides a default ExecutorSettings for tests."""
    return ExecutorSettings()


@pytest.fixture
def session(risk_config: RiskSettings, executor_config: ExecutorSettings) -> TradingSession:
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
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr_at_entry=10.0)
    session.add_trade(trade)

    assert session.trades_today == 1
    assert "order1" in session.active_trades
    assert session.active_trades["order1"] == trade


def test_finalize_winning_trade(session: TradingSession):
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade(order_id="order1", exit_price=220.0)

    assert not session.active_trades
    assert len(session.trade_history) == 1
    assert session.daily_pnl == ((220.0 - 200.0) * 50)
    assert session.current_equity == 100_000.0 + session.daily_pnl
    assert session.consecutive_losses == 0


def test_finalize_losing_trade_and_consecutive_losses(session: TradingSession):
    # First loss
    trade1 = Trade("NIFTY_CE", "BUY", 200.0, 50, "order1", atr_at_entry=10.0)
    session.add_trade(trade1)
    session.finalize_trade(order_id="order1", exit_price=190.0)
    assert session.consecutive_losses == 1
    assert session.daily_pnl == -500.0

    # Second loss
    trade2 = Trade("NIFTY_PE", "BUY", 150.0, 50, "order2", atr_at_entry=8.0)
    session.add_trade(trade2)
    session.finalize_trade(order_id="order2", exit_price=145.0)
    assert session.consecutive_losses == 2
    assert session.daily_pnl == -750.0

    # A winning trade should reset the counter
    trade3 = Trade("NIFTY_CE", "BUY", 200.0, 50, "order3", atr_at_entry=10.0)
    session.add_trade(trade3)
    session.finalize_trade(order_id="order3", exit_price=210.0)
    assert session.consecutive_losses == 0


def test_risk_limit_max_trades(session: TradingSession):
    # We allow exactly 5 trades; after the 5th is recorded, limits should report a breach
    for i in range(5):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr_at_entry=10.0)
        session.add_trade(trade)
        if i < 4:
            assert session.check_risk_limits() is None
        else:
            # After adding the 5th trade, the session should report the limit reached
            assert session.check_risk_limits() is not None
        session.finalize_trade(f"order{i}", 201.0)

    assert session.trades_today == 5
    assert session.check_risk_limits() is not None  # block before a 6th trade


def test_risk_limit_consecutive_loss(session: TradingSession):
    for i in range(2):  # 2 losses; limit is 3
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr_at_entry=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 199.0)
        assert session.check_risk_limits() is None

    # 3rd loss hits the limit
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order3", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade("order3", 199.0)
    assert session.consecutive_losses == 3
    assert session.check_risk_limits() is not None  # Limit reached


def test_risk_limit_max_drawdown(risk_config: RiskSettings, executor_config: ExecutorSettings):
    # Increase consecutive loss limit to isolate drawdown behavior
    risk_config.consecutive_loss_limit = 10
    risk_config.max_trades_per_day = 10
    session = TradingSession(risk_config=risk_config, executor_config=executor_config, starting_equity=100_000.0)

    # 5% of 100k is 5k
    # Each trade loses (200 - 180) * 50 = 1000. So, 4 trades → 4000 loss (< 5k)
    for i in range(4):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 50, f"order{i}", atr_at_entry=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 180.0)

    assert session.drawdown_pct < session.risk_config.max_daily_drawdown_pct
    assert session.check_risk_limits() is None, "Risk limit should not be hit yet"

    # The 5th trade should breach the drawdown limit
    trade = Trade("NIFTY_CE", "BUY", 200.0, 50, "order5", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade("order5", 180.0)

    assert session.drawdown_pct >= session.risk_config.max_daily_drawdown_pct
    result = session.check_risk_limits()
    assert result is not None, "Drawdown risk limit should be hit"
    assert "Max daily drawdown" in result
