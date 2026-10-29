import pytest

from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine
from nifty_scalper_bot.risk.cost_model import estimate_order_cost


class Hub:
    def get_quote(self, symbol, allow_pull=False):
        return {
            "ltp": 100,
            "bid": 99.95,
            "ask": 100.05,
            "ask_quantity": 65,
            "bid_quantity": 65,
        }


class Resolver:
    def lot_size_for_symbol(self, symbol):
        return 65


def test_limit_partial_fill_charges_one_order_brokerage_and_respects_depth():
    engine = PaperFillEngine(Hub(), Resolver())
    order = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 130,
            "order_type": "LIMIT",
            "price": 101,
        }
    )
    assert order["filled_quantity"] == 65
    assert order["average_price"] == 100.05
    assert order["fees"] == estimate_order_cost(turnover=65 * 100.05, side="BUY")
    engine.process_quote(order["symbol"])
    final = engine.get_orders()[0]
    assert final["filled_quantity"] == 130
    assert final["average_price"] == 100.05
    assert final["fees"] == estimate_order_cost(
        turnover=130 * 100.05,
        side="BUY",
    )


def test_slippage_cannot_fill_limit_beyond_its_price(monkeypatch):
    monkeypatch.setenv("PAPER__SLIPPAGE_BPS", "50")
    engine = PaperFillEngine(Hub(), Resolver())
    order = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "order_type": "LIMIT",
            "price": 100.10,
        }
    )
    assert order["filled_quantity"] == 0
    assert order["fees"] == 0


def test_limit_queue_ahead_is_isolated_per_order(monkeypatch):
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "5")
    engine = PaperFillEngine(Hub(), Resolver())

    first = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "order_type": "LIMIT",
            "price": 101,
        }
    )
    second = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "order_type": "LIMIT",
            "price": 101,
        }
    )

    assert first["filled_quantity"] == 60
    assert second["filled_quantity"] == 60
    assert first["queue_ahead"] == 0
    assert second["queue_ahead"] == 0


def test_limit_price_change_resets_queue_priority(monkeypatch):
    monkeypatch.setenv("PAPER__QUEUE_DEPTH", "5")
    engine = PaperFillEngine(Hub(), Resolver())
    order = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "order_type": "LIMIT",
            "price": 101,
        }
    )
    assert order["filled_quantity"] == 60

    modified = engine.modify_order(order["order_id"], price=101.5)

    assert modified["filled_quantity"] == 60
    assert modified["queue_ahead"] == 0
    assert modified["remaining_quantity"] == 5


def test_calibrated_limit_order_waits_for_quote_and_amendment_restarts_latency():
    engine = PaperFillEngine(Hub(), Resolver())
    now = [1_000.0]
    engine.set_clock(lambda: now[0])
    engine.calibrate_from_completed_trades(
        [{"execution_quality": {"entry_submit_to_fill_seconds": 0.5}}]
    )
    placed = engine.place_order(
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "order_type": "LIMIT",
            "price": 101,
        }
    )
    assert placed["status"] == "open"
    assert placed["filled_quantity"] == 0

    now[0] = 1_000.3
    modified = engine.modify_order(placed["order_id"], price=101.5)
    assert modified["filled_quantity"] == 0
    now[0] = 1_000.5
    engine.process_quote(placed["symbol"])
    assert engine.get_orders()[0]["status"] == "open"

    now[0] = 1_000.8
    engine.process_quote(placed["symbol"])
    completed = engine.get_orders()[0]
    assert completed["status"] == "complete"
    assert completed["average_price"] == 100.05
    assert completed["fill_latency_seconds"] == pytest.approx(0.5)
    assert completed["first_fill_timestamp"] == pytest.approx(1_000.8)
