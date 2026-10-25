"""Broker-order evidence must match every protective stop invariant."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.bracket_core import BracketManager
from nifty_scalper_bot.execution.exchange_stop_proof import (
    verify_exchange_stop_order,
)


def _valid_order() -> dict:
    return {
        "order_id": "stop-123",
        "status": "TRIGGER PENDING",
        "exchange": "NFO",
        "tradingsymbol": "NIFTY26OCT22650PE",
        "transaction_type": "SELL",
        "product": "MIS",
        "order_type": "SL",
        "quantity": 65,
        "pending_quantity": 65,
        "filled_quantity": 0,
        "trigger_price": 109.2,
        "price": 103.75,
    }


def _verify(evidence: dict) -> tuple[bool, str | None]:
    return verify_exchange_stop_order(
        evidence,
        order_id="stop-123",
        symbol="NFO:NIFTY26OCT22650PE",
        entry_side="BUY",
        product="MIS",
        quantity=65,
        trigger_price=109.2,
        limit_price=103.75,
    )


@pytest.mark.parametrize("status", ["OPEN", "TRIGGER PENDING"])
def test_exact_broker_orderbook_identity_proves_stop(status: str) -> None:
    order = _valid_order()
    order["status"] = status
    assert _verify(order) == (True, None)


@pytest.mark.parametrize(
    "key,bad,reason",
    [
        ("order_id", "other", "broker_stop_order_id_mismatch"),
        ("order_id", None, "broker_stop_order_id_mismatch"),
        ("tradingsymbol", "NIFTY26OCT22600PE", "broker_stop_symbol_mismatch"),
        ("tradingsymbol", None, "broker_stop_symbol_mismatch"),
        ("exchange", "NSE", "broker_stop_exchange_unverified"),
        ("exchange", None, "broker_stop_exchange_unverified"),
        ("transaction_type", "BUY", "broker_stop_side_mismatch"),
        ("transaction_type", None, "broker_stop_side_mismatch"),
        ("product", "NRML", "broker_stop_product_mismatch"),
        ("product", None, "broker_stop_product_mismatch"),
        ("order_type", "LIMIT", "broker_stop_order_type_unverified"),
        ("order_type", None, "broker_stop_order_type_unverified"),
        ("quantity", 64, "broker_stop_quantity_mismatch"),
        ("quantity", 65.5, "broker_stop_quantity_mismatch"),
        ("quantity", None, "broker_stop_quantity_mismatch"),
        ("pending_quantity", 50, "broker_stop_remaining_quantity_mismatch"),
        ("filled_quantity", 15, "broker_stop_partially_filled"),
        ("trigger_price", 100.0, "broker_stop_trigger_mismatch"),
        ("trigger_price", None, "broker_stop_trigger_mismatch"),
        ("price", 104.0, "broker_stop_limit_mismatch"),
        ("price", 109.2, "broker_stop_limit_mismatch"),
        ("price", None, "broker_stop_limit_mismatch"),
        ("status", "MODIFY PENDING", "broker_stop_status_unverified:MODIFY PENDING"),
        ("status", "CANCELLED", "broker_stop_status_unverified:CANCELLED"),
    ],
)
def test_wrong_or_unproven_broker_order_never_counts_as_protection(
    key: str, bad: object, reason: str
) -> None:
    order = _valid_order()
    order[key] = bad
    assert _verify(order) == (False, reason)


def test_nonstop_limit_order_cannot_masquerade_as_protection() -> None:
    fake = _valid_order()
    fake["order_type"] = "LIMIT"
    assert _verify(fake)[0] is False


def test_modified_stop_remains_unverified_until_trigger_matches_broker() -> None:
    before_modify = _valid_order()
    assert _verify(before_modify) == (True, None)
    modified_expectation = {
        "order_id": "stop-123",
        "symbol": "NFO:NIFTY26OCT22650PE",
        "entry_side": "BUY",
        "product": "MIS",
        "quantity": 65,
        "trigger_price": 120.65,
        "limit_price": 114.6,
    }
    assert verify_exchange_stop_order(before_modify, **modified_expectation) == (
        False,
        "broker_stop_trigger_mismatch",
    )
    acknowledged = deepcopy(before_modify)
    acknowledged["trigger_price"] = 120.65
    acknowledged["price"] = 114.6
    assert verify_exchange_stop_order(acknowledged, **modified_expectation) == (
        True,
        None,
    )


def test_protective_buy_sl_market_can_be_verified() -> None:
    order = _valid_order()
    order.update(
        transaction_type="BUY",
        order_type="SL-M",
        trigger_price=109.2,
        price=0,
    )
    confirmed, reason = verify_exchange_stop_order(
        order,
        order_id="stop-123",
        symbol="NFO:NIFTY26OCT22650PE",
        entry_side="SELL",
        product="MIS",
        quantity=65,
        trigger_price=109.2,
        limit_price=None,
    )
    assert confirmed and reason is None


def test_core_confirmation_latches_mismatched_broker_identity() -> None:
    """Integration: incorrect orderbook snapshot must block later LIVE entries."""
    manager = object.__new__(BracketManager)
    manager._lock = RLock()
    order = _valid_order()
    order["transaction_type"] = "BUY"
    manager._get_broker_order_status = lambda order_id: order
    bracket = SimpleNamespace(
        exchange_stop_order_id="stop-123",
        exchange_stop_quantity=65,
        exchange_stop_trigger_price=109.2,
        exchange_stop_limit_price=103.75,
        exchange_stop_degraded_reason=None,
        symbol="NFO:NIFTY26OCT22650PE",
        side="BUY",
        product="MIS",
        updated_at=0.0,
    )
    assert manager._confirm_exchange_protective_stop(bracket) is False
    assert bracket.exchange_stop_degraded_reason == "broker_stop_side_mismatch"
    order["transaction_type"] = "SELL"
    assert manager._confirm_exchange_protective_stop(bracket) is True
    assert bracket.exchange_stop_degraded_reason is None



def test_sparse_history_obtains_exact_live_stop_orderbook_proof() -> None:
    """Entry and restart paths share this order-evidence lookup."""
    reads = []
    manager = object.__new__(BracketManager)
    manager.order_manager = SimpleNamespace(
        _broker=SimpleNamespace(
            get_order_status=lambda _id: {"order_id": "stop-123", "status": "OPEN"},
            get_orders=lambda: reads.append("book") or [_valid_order()],
        )
    )
    assert _verify(dict(manager._get_broker_order_status("stop-123"))) == (True, None)
    assert reads == ["book"]


def test_sparse_history_missing_exact_order_cannot_confirm_protection() -> None:
    other = _valid_order()
    other["order_id"] = "another-stop"
    manager = object.__new__(BracketManager)
    manager.order_manager = SimpleNamespace(
        _broker=SimpleNamespace(
            get_order_status=lambda _id: {"order_id": "stop-123", "status": "OPEN"},
            get_orders=lambda: [other],
        )
    )
    assert _verify(dict(manager._get_broker_order_status("stop-123")))[0] is False


def test_terminal_history_does_not_use_stale_open_orderbook() -> None:
    def wrong_fallback():
        raise AssertionError("terminal broker history must win")

    manager = object.__new__(BracketManager)
    manager.order_manager = SimpleNamespace(
        _broker=SimpleNamespace(
            get_order_status=lambda order_id: {
                "order_id": order_id, "status": "CANCELLED"
            },
            get_orders=wrong_fallback,
        )
    )
    evidence = manager._get_broker_order_status("stop-123")
    assert evidence["status"] == "CANCELLED"


def test_complete_history_does_not_require_secondary_broker_read() -> None:
    manager = object.__new__(BracketManager)
    manager.order_manager = SimpleNamespace(
        _broker=SimpleNamespace(
            get_order_status=lambda _id: _valid_order(),
            get_orders=lambda: (_ for _ in ()).throw(AssertionError("not needed")),
        )
    )
    assert _verify(dict(manager._get_broker_order_status("stop-123"))) == (True, None)
