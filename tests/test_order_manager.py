from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from src.execution.order_executor import OrderManager
from src.execution.order_state import OrderSide, OrderState


class DummyBroker:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def place(self, payload: Dict[str, Any]) -> str:
        self.calls.append(payload)
        return f"O{len(self.calls)}"


def _quote() -> Dict[str, Any]:
    return {
        "bid": 99.0,
        "ask": 100.0,
        "depth": {
            "buy": [{"price": 99.0, "quantity": 10}],
            "sell": [
                {"price": 100.0, "quantity": 5},
                {"price": 101.0, "quantity": 5},
            ],
        },
        "tick": 1.0,
    }


def test_partial_fill_then_complete() -> None:
    broker = DummyBroker()
    om = OrderManager(broker.place, tick_size=1.0, fill_timeout_ms=1000)
    om.submit({"action": "BUY", "symbol": "FOO", "quantity": 8, "quote": _quote()})
    cid = next(iter(om.orders))
    om.handle_partial(cid, 3, 100.0, _quote())
    leg = om.orders[cid]
    assert leg.state == OrderState.PARTIAL
    assert len(broker.calls) == 2
    om.handle_fill(cid, 101.0)
    assert leg.state == OrderState.FILLED


def test_timeout_cancel() -> None:
    broker = DummyBroker()
    om = OrderManager(broker.place, tick_size=1.0, fill_timeout_ms=1)
    om.submit({"action": "BUY", "symbol": "FOO", "quantity": 1, "quote": _quote()})
    cid = next(iter(om.orders))
    om.orders[cid].expires_at = datetime.utcnow() - timedelta(milliseconds=10)
    om.check_timeouts()
    assert om.orders[cid].state == OrderState.CANCELLED


def test_limit_clamp_by_depth() -> None:
    broker = DummyBroker()
    om = OrderManager(broker.place, tick_size=1.0, max_slip_ticks=2)
    quote = {
        "bid": 99.0,
        "ask": 100.0,
        "depth": {
            "sell": [
                {"price": 100.0, "quantity": 2},
                {"price": 101.0, "quantity": 2},
                {"price": 102.0, "quantity": 2},
            ]
        },
        "tick": 1.0,
    }
    price, _ = om.calc_limit_price(OrderSide.BUY, 10, quote)
    assert price == 102.0


def test_calc_limit_price_uses_marketable_fallback_for_buy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = DummyBroker()
    om = OrderManager(broker.place, tick_size=0.05)

    monkeypatch.setattr(om, "get_marketable_ask", lambda symbol: 201.25)

    quote = {"ask": 0.0, "depth": {}, "tick": 0.05, "tradingsymbol": "NIFTY24"}
    price, slip = om.calc_limit_price(OrderSide.BUY, 50, quote)

    assert price == pytest.approx(201.25)
    assert slip == pytest.approx(201.25)


def test_calc_limit_price_uses_marketable_fallback_for_sell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = DummyBroker()
    om = OrderManager(broker.place, tick_size=0.05)

    monkeypatch.setattr(om, "get_marketable_ask", lambda symbol: 201.25)

    quote = {"bid": 0.0, "depth": {}, "tick": 0.05, "tradingsymbol": "NIFTY24"}
    price, slip = om.calc_limit_price(OrderSide.SELL, 50, quote)

    assert price == pytest.approx(201.20)
    assert slip == pytest.approx(201.20)
