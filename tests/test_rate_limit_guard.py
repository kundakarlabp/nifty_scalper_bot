from unittest.mock import Mock

from src.execution.order_executor import OrderExecutor


def test_rate_limit_blocks_excess_orders() -> None:
    kite = Mock()
    kite.place_order.return_value = "OID"
    ex = OrderExecutor(kite=kite, telegram_controller=None)
    ex.max_orders_per_min = 2
    res1 = ex._place_with_cb({})
    res2 = ex._place_with_cb({})
    res3 = ex._place_with_cb({})
    assert res1["ok"] and res2["ok"]
    assert not res3["ok"]
    assert res3["reason"] == "rate_limited"
    assert kite.place_order.call_count == 2
