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
from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

SYMBOL = "NFO:NIFTY26NOV25700CE"


class _Broker:
    is_simulated_adapter = True

    def __init__(self, quantity: int = 65) -> None:
        self.quantity = quantity
        self.position_product = "MIS"
        self.orders: list[dict] = []
        self.reply: dict = {"order_id": "new-order", "status": "SUBMITTED"}
        self.position_outage = False

    def get_positions(self):
        if self.position_outage:
            raise RuntimeError("broker outage")
        return {
            "net": [
                {
                    "tradingsymbol": SYMBOL,
                    "quantity": self.quantity,
                    "product": self.position_product,
                }
            ]
        }

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

    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT", "MIS")

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

    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT", "MIS")

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

    assert manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT", "MIS") is None


def test_unknown_contract_lot_blocks_live_exit_without_proven_entry_lot(
    monkeypatch, tmp_path
) -> None:
    manager, _broker = _manager(monkeypatch, tmp_path)

    def offline(_symbol):
        raise OrderPlacementError("lot_size_unresolved")

    monkeypatch.setattr(manager, "_lot_size_for_symbol", offline)
    result = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT", "MIS")

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


def test_live_exit_rejects_mismatched_broker_product_in_both_routes(
    monkeypatch, tmp_path
) -> None:
    manager, broker = _manager(monkeypatch, tmp_path)
    broker.position_product = "NRML"
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    blocked = manager._validate_option_exit_quantity(SYMBOL, "SELL", 65, "EXIT", "MIS")
    assert blocked is not None
    assert blocked["reason"] == "exit_broker_product_mismatch"
    assert blocked["broker_product"] == "NRML"

    with pytest.raises(OrderPlacementError, match="exit_broker_product_mismatch"):
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


def test_multi_product_snapshot_preserves_each_signed_broker_exposure() -> None:
    snapshot = decode_position_snapshot(
        {
            "net": [
                {"tradingsymbol": SYMBOL, "quantity": 65, "product": "MIS"},
                {"tradingsymbol": SYMBOL, "quantity": -65, "product": "NRML"},
            ]
        }
    )
    assert snapshot.all_flat is False
    assert snapshot.quantity_for(SYMBOL, product="MIS") == 65
    assert snapshot.quantity_for(SYMBOL, product="NRML") == -65
    with pytest.raises(PositionSnapshotError, match="product ambiguous"):
        snapshot.quantity_for(SYMBOL)


def test_live_exit_uses_exact_product_not_cross_product_net(monkeypatch, tmp_path):
    manager, broker = _manager(monkeypatch, tmp_path)
    broker.get_positions = lambda: {
        "net": [
            {"tradingsymbol": SYMBOL, "quantity": 65, "product": "MIS"},
            {"tradingsymbol": SYMBOL, "quantity": -65, "product": "NRML"},
        ]
    }
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    assert manager._validate_option_exit_quantity(
        SYMBOL, "SELL", 65, "EXIT", "MIS"
    ) is None
    assert manager._validate_option_exit_quantity(
        SYMBOL, "BUY", 65, "EXIT", "NRML"
    ) is None
    assert manager._validate_option_exit_quantity(
        SYMBOL, "SELL", 130, "EXIT", "MIS"
    )["reason"] == "exit_quantity_exceeds_position"


def test_live_exit_refuses_broker_position_missing_product(monkeypatch, tmp_path):
    manager, broker = _manager(monkeypatch, tmp_path)
    broker.position_product = ""
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    blocked = manager._validate_option_exit_quantity(
        SYMBOL, "SELL", 65, "EXIT", "MIS"
    )
    assert blocked is not None
    assert blocked["reason"] == "exit_broker_product_unavailable"

    with pytest.raises(OrderPlacementError, match="exit_broker_product_unavailable"):
        manager._place_exit_order(
            symbol=SYMBOL,
            side="SELL",
            quantity=65,
            product="MIS",
            tag="FORCE_SL_EXIT",
        )
    assert not broker.orders


def test_multi_product_snapshot_rejects_missing_or_duplicate_identity() -> None:
    with pytest.raises(PositionSnapshotError):
        decode_position_snapshot(
            [
                {"symbol": SYMBOL, "quantity": 65, "product": "MIS"},
                {"symbol": SYMBOL, "quantity": 0},
            ]
        )
    with pytest.raises(PositionSnapshotError, match="duplicate broker position"):
        decode_position_snapshot(
            [
                {"symbol": SYMBOL, "quantity": 65, "product": "MIS"},
                {"symbol": SYMBOL, "quantity": 0, "product": "MIS"},
            ]
        )


def test_legacy_exit_uses_the_native_durable_submission_owner(
    monkeypatch, tmp_path
) -> None:
    import nifty_scalper_bot.execution.order_manager_core as core

    manager, _broker = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    calls: list[dict] = []
    manager._orders["native-exit"] = SimpleNamespace(order_id="native-exit")

    def native_place(**kwargs):
        calls.append(kwargs)
        return "native-exit"

    def reject_parallel_path(**_kwargs):
        pytest.fail("legacy exit must not bypass the native owner")

    monkeypatch.setattr(manager, "place_order", native_place)
    monkeypatch.setattr(manager, "_place_single_order", reject_parallel_path)

    def route(**kwargs):
        ok, order_id, _error = kwargs["submit"]("MIS", "IOC", "regular")
        return SimpleNamespace(
            ok=ok,
            order_id=order_id,
            reason=None,
            attempts=[SimpleNamespace(attempt=1)],
        )

    monkeypatch.setattr(core, "plan_and_send_exit", route)
    details = manager._place_exit_order(
        symbol=SYMBOL, side="SELL", quantity=65, product="MIS", tag="FORCE_SL_EXIT"
    )

    assert details.order_id == "native-exit"
    assert len(calls) == 1
    assert calls[0]["intent"] == "EXIT"
    assert calls[0]["check_risk"] is False
    assert calls[0]["product"] == "MIS"
    assert calls[0]["quantity"] == 65
