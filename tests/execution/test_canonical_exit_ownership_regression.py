"""Regression checks for one broker-exit owner and exit-admission separation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

SYMBOL = "NFO:NIFTY26NOV25700CE"


class _Broker:
    is_simulated_adapter = True

    def __init__(self) -> None:
        self.submissions: list[dict] = []

    def place_order(self, **kwargs):
        self.submissions.append(dict(kwargs))
        return {"order_id": f"exit-{len(self.submissions)}", "status": "SUBMITTED"}


def _order_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _Broker()
    manager = OrderManager(
        broker_client=broker,
        position_manager=PositionManager(str(tmp_path / "positions.json")),
        rate_limiter=RateLimiter(),
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    return manager, broker


@pytest.mark.parametrize("intent", ["EXIT", "REDUCE"])
def test_missing_bid_ask_cannot_block_canonical_reducing_order(
    monkeypatch, tmp_path, intent
) -> None:
    manager, broker = _order_manager(monkeypatch, tmp_path)
    manager._positions.open_position(SYMBOL, "LONG", 65, 145.0, order_id="entry-1")
    # Exercise precisely the selected-option LIVE quote admission condition.
    # Keep the broker adapter simulated: this must never place a real order.
    monkeypatch.setattr(manager, "_execution_mode_env", lambda: "LIVE")
    monkeypatch.setattr(
        manager, "_is_selected_option_for_live_execution", lambda _symbol: True
    )
    monkeypatch.setattr(
        manager, "_get_latest_quote_safe", lambda _symbol: {"ltp": 140.0}
    )

    result = manager.place_order(
        symbol=SYMBOL,
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent=intent,
        tag="EXIT_HARD_SL",
    )

    assert result == "exit-1"
    assert len(broker.submissions) == 1
    assert broker.submissions[0]["quantity"] == 65


def test_missing_bid_ask_still_blocks_new_selected_option_entry(
    monkeypatch, tmp_path
) -> None:
    manager, broker = _order_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "_execution_mode_env", lambda: "LIVE")
    monkeypatch.setattr(
        manager, "_is_selected_option_for_live_execution", lambda _symbol: True
    )
    monkeypatch.setattr(manager, "_get_latest_quote_safe", lambda _symbol: {})

    result = manager.place_order(
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=145.0,
        stop_loss=139.0,
        take_profit=160.0,
        check_risk=False,
        intent="ENTRY",
        signal_id="entry-without-depth",
    )

    assert result is None
    assert not broker.submissions
    assert manager._last_order_decision["block_reason"] == (
        "selected_option_bid_ask_missing"
    )


def test_canonical_bracket_owns_stop_even_when_local_stop_is_crossed(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _order_manager(monkeypatch, tmp_path)
    position = manager._positions.open_position(
        SYMBOL, "LONG", 65, 145.0, stop_loss=140.0, order_id="entry-1"
    )
    manager._data_hub = SimpleNamespace(get_latest_price=lambda _symbol: 130.0)
    manager._bracket_manager = SimpleNamespace(register_virtual_bracket=lambda **_: None)
    attempts: list[dict] = []
    monkeypatch.setattr(
        manager, "_place_exit_order", lambda **kwargs: attempts.append(kwargs)
    )

    manager._check_force_stop_losses()

    assert not attempts, "canonical bracket owns every protected stop"
    assert position.state != "force_closed_by_sl"


def test_legacy_stop_submission_failure_is_retryable_not_marked_closed(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _order_manager(monkeypatch, tmp_path)
    position = manager._positions.open_position(
        SYMBOL, "LONG", 65, 145.0, stop_loss=140.0, order_id="entry-1"
    )
    manager._data_hub = SimpleNamespace(get_latest_price=lambda _symbol: 130.0)
    attempts: list[dict] = []

    def reject(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(manager, "_place_exit_order", reject)
    manager._check_force_stop_losses()
    manager._check_force_stop_losses()

    assert len(attempts) == 2
    assert position.state != "force_closed_by_sl"


def test_legacy_stop_records_acknowledged_exit_not_filled_position(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _order_manager(monkeypatch, tmp_path)
    position = manager._positions.open_position(
        SYMBOL, "LONG", 65, 145.0, stop_loss=140.0, order_id="entry-1"
    )
    manager._data_hub = SimpleNamespace(get_latest_price=lambda _symbol: 130.0)
    attempts: list[dict] = []

    def submit(**kwargs):
        attempts.append(kwargs)
        return SimpleNamespace(order_id="legacy-1")

    monkeypatch.setattr(manager, "_place_exit_order", submit)
    manager._check_force_stop_losses()
    manager._check_force_stop_losses()

    assert len(attempts) == 1
    assert position.state == "force_exit_submitted"
