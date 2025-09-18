from unittest.mock import Mock

from src.execution.order_executor import NetworkException, OrderExecutor


def test_place_with_retry_and_breaker() -> None:
    kite = Mock()
    kite.place_order.side_effect = [
        NetworkException("timeout"),
        NetworkException("HTTP 502"),
        "OID123",
    ]
    ex = OrderExecutor(kite=kite, telegram_controller=None)
    res = ex._place_with_cb({})
    assert res["ok"]
    assert res["order_id"] == "OID123"
    assert kite.place_order.call_count == 3
    assert ex.cb_orders.health()["n"] >= 3
    assert res["attempts"] == 3
    assert res["retries"] == 2
    assert isinstance(res["retry_delays_ms"], list)
    assert len(res["retry_delays_ms"]) == 2

    ex.cb_orders.force_open(30)
    kite.place_order.reset_mock()
    res2 = ex._place_with_cb({})
    assert not res2["ok"]
    assert res2["reason"] == "api_breaker_open"
    assert kite.place_order.call_count == 0
    assert res2["attempts"] == 1
    assert res2["retries"] == 0
    assert res2["retry_delays_ms"] is None
