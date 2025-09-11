import pytest
from src.execution.order_executor import OrderExecutor


def test_price_with_slippage():
    oe = OrderExecutor(kite=None)
    buy = oe._price_with_slippage(100.0, "BUY", 0.25)
    sell = oe._price_with_slippage(100.0, "SELL", 0.25)
    assert buy == pytest.approx(100.0 * 1.0025)
    assert sell == pytest.approx(100.0 * 0.9975)
