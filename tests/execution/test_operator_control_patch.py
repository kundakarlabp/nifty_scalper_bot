from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.execution import OrderManager
from nifty_scalper_bot.execution.operator_control_patch import emergency_stop, flatten_all


@pytest.fixture(autouse=True)
def _reset_trading_switch() -> None:
    trading_switch().resume()
    yield
    trading_switch().resume()


def test_runtime_order_manager_exposes_emergency_and_flatten_controls() -> None:
    assert hasattr(OrderManager, "emergency_stop")
    assert hasattr(OrderManager, "engage_kill_switch")
    assert hasattr(OrderManager, "kill_switch")
    assert hasattr(OrderManager, "flatten_all")
    assert hasattr(OrderManager, "flatten_positions")
    assert hasattr(OrderManager, "close_all_positions")


def test_emergency_stop_latches_kill_switch_and_cancels_open_orders() -> None:
    cancelled: list[str] = []
    manager = SimpleNamespace(
        _logger=SimpleNamespace(warning=lambda *_a, **_k: None, critical=lambda *_a, **_k: None),
        get_open_orders=lambda: [{"order_id": "OID1", "status": "OPEN"}],
        cancel_order=lambda order_id: cancelled.append(order_id),
    )

    result = emergency_stop(manager, reason="test")

    assert result["kill_switch"] is True
    assert result["cancelled"] == ["OID1"]
    assert cancelled == ["OID1"]
    assert getattr(manager, "_kill_switch_reason") == "test"


def test_flatten_all_cancels_pending_and_sends_market_exit_for_open_position() -> None:
    sent: list[dict] = []
    manager = SimpleNamespace(
        _logger=SimpleNamespace(warning=lambda *_a, **_k: None, critical=lambda *_a, **_k: None),
        get_open_orders=lambda: [{"order_id": "OPEN1", "status": "OPEN"}],
        cancel_order=lambda _order_id: None,
        _broker=SimpleNamespace(get_positions=lambda: [{"tradingsymbol": "NFO:NIFTY2671424250CE", "quantity": 65}]),
    )

    def _place_order(**kwargs):
        sent.append(kwargs)
        return "EXIT1"

    manager.place_order = _place_order

    result = flatten_all(manager, reason="test_flat")

    assert result["submitted"] == [
        {"symbol": "NFO:NIFTY2671424250CE", "qty": 65, "side": "SELL", "order_id": "EXIT1"}
    ]
    assert sent[0]["side"] == "SELL"
    assert sent[0]["order_type"] == "MARKET"
    assert sent[0]["check_risk"] is False
