from __future__ import annotations

from dataclasses import dataclass

import pytest

from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


@dataclass
class DummyResolver:
    lot_size: int = 1

    def lot_size_for_symbol(self, symbol: str) -> int:
        return self.lot_size


@dataclass
class DummyHub:
    quote: dict[str, float]

    def get_quote(self, symbol: str, allow_pull: bool = True) -> dict[str, float]:
        return dict(self.quote)


@pytest.fixture(autouse=True)
def reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAPER__QUEUE_DEPTH", raising=False)
    monkeypatch.delenv("PAPER__SLIPPAGE_BPS", raising=False)
    monkeypatch.delenv("PAPER__RESPECT_LOTSIZE", raising=False)


def test_market_order_slippage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "0")
    hub = DummyHub(
        {
            "best_bid": 100.0,
            "best_ask": 100.5,
            "ltp": 100.25,
            "close": 99.0,
        }
    )
    engine = PaperFillEngine(hub, DummyResolver())

    order = engine.place_order(
        {"symbol": "TEST", "side": "BUY", "quantity": 10, "order_type": "MARKET"}
    )

    assert order["status"] == "complete"
    assert order["filled_quantity"] == 10
    assert order["average_price"] > 100.25


def test_limit_order_partial_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "2")
    hub = DummyHub({"best_bid": 100.0, "best_ask": 100.2})
    engine = PaperFillEngine(hub, DummyResolver())

    order = engine.place_order(
        {
            "symbol": "TEST",
            "side": "BUY",
            "quantity": 5,
            "order_type": "LIMIT",
            "price": 100.3,
        }
    )

    assert order["status"] == "open"
    assert order["filled_quantity"] == 3
    assert order["remaining_quantity"] == 2


def test_cancel_cascades_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "5")
    hub = DummyHub({"best_bid": 100.0, "best_ask": 100.1})
    engine = PaperFillEngine(hub, DummyResolver())

    parent = engine.place_order(
        {
            "symbol": "TEST",
            "side": "BUY",
            "quantity": 2,
            "order_type": "LIMIT",
            "price": 100.2,
        }
    )
    child = engine.place_order(
        {
            "symbol": "TEST",
            "side": "SELL",
            "quantity": 2,
            "order_type": "STOP_LOSS",
            "price": 99.5,
            "parent_order_id": parent["order_id"],
        }
    )

    cancelled = engine.cancel_order(parent["order_id"])

    assert cancelled["status"] == "cancelled"
    child_state = engine.orders[child["order_id"]]
    assert child_state["status"] == "cancelled"
    assert child_state["message"] == "Parent cancelled"


def test_slippage_scales_with_order_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "0")
    base_quote = {
        "best_bid": 100.0,
        "best_ask": 100.2,
        "ltp": 100.1,
        "close": 99.5,
    }
    small_engine = PaperFillEngine(DummyHub(base_quote), DummyResolver())
    large_engine = PaperFillEngine(DummyHub(base_quote), DummyResolver())

    small_order = small_engine.place_order(
        {
            "symbol": "NIFTY25FEB24000CE",
            "side": "BUY",
            "quantity": 25,
            "order_type": "MARKET",
        }
    )
    large_order = large_engine.place_order(
        {
            "symbol": "NIFTY25FEB24000CE",
            "side": "BUY",
            "quantity": 150,
            "order_type": "MARKET",
        }
    )

    assert large_order["average_price"] > small_order["average_price"]
