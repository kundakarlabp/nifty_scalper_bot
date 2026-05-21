from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.errors import OrderPlacementError


@pytest.fixture()
def runner_with_mocks() -> StrategyRunner:
    market_data = MagicMock()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    position_manager = MagicMock()

    risk_manager.suggest_position_size.return_value = 10
    risk_manager.validate_new_position.return_value = (False, "Cooldown active for 1s")
    order_manager.consume_skip_reason.return_value = None

    cfg = StrategyRunnerConfig(max_trade_history=4)
    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        message_bus=MagicMock(),
        config=cfg,
        data_hub=None,
        strike_selector=None,
    )
    runner.add_symbol("NIFTY")
    return runner


def test_runner_logs_code_not_text(
    runner_with_mocks: StrategyRunner, caplog: pytest.LogCaptureFixture
) -> None:
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=5,
        confidence=1.0,
        reason="test",
        stop_loss=98.0,
        take_profit=105.0,
        metadata={"option_type": "CE"},
    )

    with caplog.at_level("INFO"):
        runner_with_mocks._handle_signal(
            signal,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
        )

    assert any("COOLDOWN" in record.getMessage() for record in caplog.records)
    assert all(
        "Cooldown active" not in record.getMessage() for record in caplog.records
    )


def test_runner_logs_stale_once(
    runner_with_mocks: StrategyRunner, caplog: pytest.LogCaptureFixture
) -> None:
    runner_with_mocks._risk_manager.validate_new_position.return_value = (
        False,
        "STALE quote",
    )
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=5,
        confidence=1.0,
        reason="test",
        stop_loss=98.0,
        take_profit=105.0,
        metadata={"option_type": "CE"},
    )

    with caplog.at_level("INFO"):
        runner_with_mocks._handle_signal(
            signal,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
        )

    risk_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Risk rejected")
    ]
    assert len(risk_logs) == 1
    assert "STALE" in risk_logs[0]


def test_runner_records_skip_reason_from_order_error(
    runner_with_mocks: StrategyRunner,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_with_mocks._risk_manager.validate_new_position.return_value = (True, "")
    runner_with_mocks._order_manager.place_order.side_effect = OrderPlacementError(
        "no_reference_price"
    )
    mock_counter_runner = MagicMock()
    mock_counter_runner.labels.return_value.inc = MagicMock()
    mock_counter_metrics = MagicMock()
    mock_counter_metrics.labels.return_value.inc = MagicMock()
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner._STRATEGY_SKIP_COUNTER",
        mock_counter_runner,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.metrics.strategy_skips_total",
        mock_counter_metrics,
    )
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=5,
        confidence=1.0,
        reason="test",
        stop_loss=98.0,
        take_profit=105.0,
        metadata={"option_type": "CE"},
    )

    with caplog.at_level("INFO"):
        runner_with_mocks._handle_signal(
            signal,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
        )

    skip_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "skip_order"
        and record.__dict__.get("reason") == "no_reference_price"
    ]
    assert skip_logs
    mock_counter_runner.labels.assert_called_with(reason="no_reference_price")
    mock_counter_runner.labels.return_value.inc.assert_called_once()
    mock_counter_metrics.labels.assert_called_with(reason="no_reference_price")
    mock_counter_metrics.labels.return_value.inc.assert_called_once()
    history = runner_with_mocks._symbol_state["NIFTY"].trade_history
    assert not history


def test_runner_records_margin_skip_reason(
    runner_with_mocks: StrategyRunner,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_with_mocks._risk_manager.validate_new_position.return_value = (True, "")
    runner_with_mocks._order_manager.place_order.side_effect = OrderPlacementError(
        "margin_no_qty"
    )
    mock_counter_runner = MagicMock()
    mock_counter_runner.labels.return_value.inc = MagicMock()
    mock_counter_metrics = MagicMock()
    mock_counter_metrics.labels.return_value.inc = MagicMock()
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner._STRATEGY_SKIP_COUNTER",
        mock_counter_runner,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.metrics.strategy_skips_total",
        mock_counter_metrics,
    )
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=5,
        confidence=1.0,
        reason="test",
        stop_loss=98.0,
        take_profit=105.0,
        metadata={"option_type": "CE"},
    )

    with caplog.at_level("INFO"):
        runner_with_mocks._handle_signal(
            signal,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
        )

    skip_logs = [
        record
        for record in caplog.records
        if record.getMessage() == "skip_order"
        and record.__dict__.get("reason") == "margin_no_qty"
    ]
    assert skip_logs
    mock_counter_runner.labels.assert_called_with(reason="margin_no_qty")
    mock_counter_runner.labels.return_value.inc.assert_called_once()
    mock_counter_metrics.labels.assert_called_with(reason="margin_no_qty")
    mock_counter_metrics.labels.return_value.inc.assert_called_once()
    history = runner_with_mocks._symbol_state["NIFTY"].trade_history
    assert not history
