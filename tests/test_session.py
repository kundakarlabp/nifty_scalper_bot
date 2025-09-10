"""
Tests for the TradingSession state manager.
"""

import pytest
from src.config import RiskSettings
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
def session(risk_config: RiskSettings) -> TradingSession:
    """Provides a TradingSession with 100k equity and lot size of 75."""
    return TradingSession(risk_config=risk_config, starting_equity=100_000.0, lot_size=75)


def test_session_initialization(session: TradingSession):
    assert session.start_equity == 100_000.0
    assert session.current_equity == 100_000.0
    assert session.daily_pnl == 0.0
    assert session.consecutive_losses == 0
    assert session.trades_today == 0
    assert not session.active_trades
    assert not session.trade_history


def test_add_trade(session: TradingSession):
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "order1", atr_at_entry=10.0)
    session.add_trade(trade)

    assert session.trades_today == 1
    assert "order1" in session.active_trades
    assert session.active_trades["order1"] == trade


def test_finalize_winning_trade(session: TradingSession):
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "order1", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade(order_id="order1", exit_price=220.0)

    assert not session.active_trades
    assert len(session.trade_history) == 1
    assert session.daily_pnl == ((220.0 - 200.0) * 75) - 20  # P&L minus fees
    assert session.current_equity == 100_000.0 + session.daily_pnl
    assert session.consecutive_losses == 0


def test_finalize_losing_trade_and_consecutive_losses(session: TradingSession):
    # First loss
    trade1 = Trade("NIFTY_CE", "BUY", 200.0, 75, "order1", atr_at_entry=10.0)
    session.add_trade(trade1)
    session.finalize_trade(order_id="order1", exit_price=190.0)
    assert session.consecutive_losses == 1
    assert session.daily_pnl == -750.0 - 20

    # Second loss
    trade2 = Trade("NIFTY_PE", "BUY", 150.0, 75, "order2", atr_at_entry=8.0)
    session.add_trade(trade2)
    session.finalize_trade(order_id="order2", exit_price=145.0)
    assert session.consecutive_losses == 2
    assert session.daily_pnl == -770.0 + (-375.0 - 20)

    # A winning trade should reset the counter
    trade3 = Trade("NIFTY_CE", "BUY", 200.0, 75, "order3", atr_at_entry=10.0)
    session.add_trade(trade3)
    session.finalize_trade(order_id="order3", exit_price=210.0)
    assert session.consecutive_losses == 0


def test_risk_limit_max_trades(session: TradingSession):
    # We allow exactly 5 trades; after the 5th is recorded, limits should report a breach
    for i in range(5):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 75, f"order{i}", atr_at_entry=10.0)
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
        trade = Trade("NIFTY_CE", "BUY", 200.0, 75, f"order{i}", atr_at_entry=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 199.0)
        assert session.check_risk_limits() is None

    # 3rd loss hits the limit
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "order3", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade("order3", 199.0)
    assert session.consecutive_losses == 3
    assert session.check_risk_limits() is not None  # Limit reached


def test_risk_limit_max_drawdown(risk_config: RiskSettings):
    # Increase consecutive loss limit to isolate drawdown behavior
    risk_config.consecutive_loss_limit = 10
    risk_config.max_trades_per_day = 20
    session = TradingSession(risk_config=risk_config, starting_equity=100_000.0, lot_size=75)

    # 5% of 100k is 5k
    # Each trade loses (200 - 180) * 75 = 1500. Net loss is 1520 including brokerage.
    # So, 3 trades → 4560 loss (< 5k)
    for i in range(3):
        trade = Trade("NIFTY_CE", "BUY", 200.0, 75, f"order{i}", atr_at_entry=10.0)
        session.add_trade(trade)
        session.finalize_trade(f"order{i}", 180.0)

    assert session.drawdown_pct < session.risk_config.max_daily_drawdown_pct
    assert session.check_risk_limits() is None, "Risk limit should not be hit yet"

    # The 4th trade should breach the drawdown limit
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "order3", atr_at_entry=10.0)
    session.add_trade(trade)
    session.finalize_trade("order3", 180.0)

    assert session.drawdown_pct >= session.risk_config.max_daily_drawdown_pct
    result = session.check_risk_limits()
    assert result is not None, "Drawdown risk limit should be hit"
    assert "Max daily drawdown" in result


def test_trade_close_warn_on_double_close(session: TradingSession, caplog: pytest.LogCaptureFixture) -> None:
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "orderX", atr_at_entry=10.0)
    trade.close(210.0)
    with caplog.at_level("WARNING"):
        trade.close(220.0)
    assert "already closed" in caplog.text


def test_trade_close_sell_branch() -> None:
    trade = Trade("NIFTY_PE", "SELL", 200.0, 75, "orderY", atr_at_entry=10.0)
    trade.close(190.0)
    assert trade.pnl == (200.0 - 190.0) * 75


def test_add_trade_duplicate(session: TradingSession, caplog: pytest.LogCaptureFixture) -> None:
    trade = Trade("NIFTY_CE", "BUY", 200.0, 75, "dup", atr_at_entry=10.0)
    session.add_trade(trade)
    with caplog.at_level("WARNING"):
        session.add_trade(trade)
    assert session.trades_today == 1
    assert len(session.active_trades) == 1
    assert "duplicate trade" in caplog.text


def test_session_requires_risksettings() -> None:
    with pytest.raises(TypeError):
        TradingSession(risk_config="bad", starting_equity=0.0)


def test_finalize_trade_missing(session: TradingSession, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert session.finalize_trade("missing", 100.0) is None
    assert "Could not find active trade" in caplog.text


def test_estimate_fees_zero_lot(session: TradingSession) -> None:
    session.lot_size = 0
    assert session._estimate_fees(100) == 0.0


def test_estimate_fees_exception(session: TradingSession) -> None:
    session.lot_size = "bad"  # type: ignore[assignment]
    assert session._estimate_fees(100) == 0.0


def test_to_status_dict(session: TradingSession) -> None:
    status = session.to_status_dict()
    assert "session_date" in status
