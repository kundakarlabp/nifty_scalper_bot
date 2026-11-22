"""LIVE option reductions require broker net inventory and valid order identities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    OrderPlacementError,
    OrderType,
)
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

SYMBOL = "NFO:NIFTY26NOV25700CE"


class _Broker:
    is_simulated_adapter = True

    def __init__(self, quantity: int = 65) -> None:
        self.quantity = quantity
        self.orders: list[dict] = []
        self.reply: dict = {"order_id": "new-order", "status": "SUBMITTED"}
        self.position_outage = False

    def get_positions(self):
        if self.position_outage:
            raise RuntimeError("broker outage")
        return {"net": [{"tradingsymbol": SYMBOL, "quantity": self.quantity}]}

    def place_order(self, **kwargs):
        self.orders.append(dict(kwargs))
        return dict(self.reply)


def _manager(monkeypatch, tmp_path, broker=None) -> tuple[OrderManager, _Broker]:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = broker or _Broker()
    manager = OrderManager(
        broker_client=broker,
        position_manager=PositionManager(str(tmp_path / "positions.json")),
        rate_limiter=RateLimiter(),
    )
    manager._positions.open_position(SYMBOL, "LONG", 65, 145.0, order_id="entry-1")
    monkeypatch.setattr(manager, "is_live_mode", lambda: True)
    return manager, broker


@pytest.mark.parametrize("broker_quantity", [0, 35, -65])
def test_live_exit_never_trusts_stale_local_65_unit_position(
    monkeypatch, tmp_path, broker_quantity
) -> None:
    manager, broker = _manager(monkeypatch, tmp_path, _Broker(quantity=broker_quantity))
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT")

    assert result is not None
    assert result["reason"] in {
        "exit_without_open_position",
        "exit_quantity_exceeds_position",
        "exit_side_not_reducing",
    }
    assert not broker.orders


def test_unavailable_broker_snapshot_is_not_treated_as_local_exposure(
    monkeypatch, tmp_path
) -> None:
    manager, broker = _manager(monkeypatch, tmp_path)
    broker.position_outage = True
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT")

    assert result is not None
    assert result["reason"] == "exit_broker_position_unavailable"


def test_trusted_entry_instrument_lot_survives_temporary_resolver_outage(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _manager(monkeypatch, tmp_path)
    manager._orders["entry-1"] = SimpleNamespace(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        intent="ENTRY",
        resolved_lot_size=65,
    )

    def offline(_symbol):
        raise OrderPlacementError("lot_size_unresolved")

    monkeypatch.setattr(manager, "_lot_size_for_symbol", offline)

    assert manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT") is None


def test_unknown_contract_lot_blocks_live_exit_without_proven_entry_lot(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _manager(monkeypatch, tmp_path)

    def offline(_symbol):
        raise OrderPlacementError("lot_size_unresolved")

    monkeypatch.setattr(manager, "_lot_size_for_symbol", offline)
    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT")

    assert result is not None
    assert result["reason"] == "exit_lot_size_unresolved"


@pytest.mark.parametrize("broker_quantity", [0, 35, -65])
def test_legacy_exit_router_cannot_bypass_live_signed_broker_position_check(
    monkeypatch, tmp_path, broker_quantity
) -> None:
    manager, broker = _manager(monkeypatch, tmp_path, _Broker(quantity=broker_quantity))
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    with pytest.raises(OrderPlacementError, match="Exit preflight blocked"):
        manager._place_exit_order(
            symbol=SYMBOL,
            side="SELL",
            quantity=65,
            product="MIS",
            tag="FORCE_SL_EXIT",
        )

    assert not broker.orders


@pytest.mark.parametrize("invalid_order_id", [None, "", "None", 0])
def test_low_level_broker_ack_requires_real_order_id(
    monkeypatch, tmp_path, invalid_order_id
) -> None:
    manager, broker = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "is_live_mode", lambda: False)
    broker.reply = {"order_id": invalid_order_id, "status": "SUBMITTED"}

    with pytest.raises(OrderPlacementError, match="missing order_id"):
        manager._place_single_order(
            symbol=SYMBOL,
            side="SELL",
            quantity=65,
            order_type=OrderType.MARKET,
            price=None,
            product="MIS",
            tag="EXIT_RESCUE",
        )

    assert len(broker.orders) == 1
    assert "None" not in manager._orders
