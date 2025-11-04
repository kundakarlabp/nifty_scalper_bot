from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.execution.order_manager import (
    ExitIntent,
    OrderDetails,
    OrderManager,
    OrderStatus,
    OrderType,
)
from nifty_scalper_bot.utils.errors import OrderPlacementError


class _StubBroker:
    def place_order(self, payload: dict) -> dict:
        return {"order_id": "oid-1", "status": "complete"}

    def get_positions(self) -> list[dict]:
        return []


class _StubRateLimiter:
    def acquire(self, *_args, **_kwargs) -> None:
        return None

    def snapshot(self) -> dict:
        return {}


def _make_order_manager() -> OrderManager:
    broker = _StubBroker()
    position_manager = MagicMock()
    limiter = _StubRateLimiter()
    om = OrderManager(broker, position_manager, limiter)
    return om


def test_place_order_enforces_lot_size(monkeypatch: pytest.MonkeyPatch) -> None:
    om = _make_order_manager()

    class _Resolver:
        def lot_size_for_symbol(self, symbol: str) -> int:
            assert symbol == "NIFTY25APR25000CE"
            return 75

    om.set_instrument_resolver(_Resolver())

    result = om.place_order(symbol="NIFTY25APR25000CE", side="BUY", quantity=74)
    assert result == ""
    assert om.consume_skip_reason() == "lot_rounding_zero"

    order_id = om.place_order(symbol="NIFTY25APR25000CE", side="BUY", quantity=150)
    assert order_id


def test_place_order_positive_quantity_required() -> None:
    om = _make_order_manager()
    om.set_instrument_resolver(MagicMock(lot_size_for_symbol=lambda _: 75))
    result = om.place_order(symbol="NIFTY25APR25000CE", side="BUY", quantity=0)
    assert result == ""
    assert om.consume_skip_reason() == "lot_rounding_zero"


def test_missing_resolver_blocks_orders() -> None:
    om = _make_order_manager()
    om.set_instrument_resolver(None)

    with pytest.raises(OrderPlacementError):
        om.place_order(symbol="NIFTY25APR25000CE", side="BUY", quantity=150)


def test_place_reduce_only_exit_caps_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _StubBroker()
    broker.get_positions = MagicMock(
        return_value=[
            {
                "tradingsymbol": "NIFTY24JUN25450CE",
                "product": "MIS",
                "quantity": 150,
            }
        ]
    )
    position_manager = MagicMock()
    position_manager.get_position.return_value = SimpleNamespace(
        side="LONG", quantity=150
    )
    limiter = _StubRateLimiter()
    order_manager = OrderManager(broker, position_manager, limiter)

    captured: dict[str, object] = {}

    def _fake_exit_order(
        *,
        symbol: str,
        side: str,
        quantity: int,
        product: str | None,
        tag: str | None,
    ) -> OrderDetails:
        captured.update(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "product": product,
                "tag": tag,
            }
        )
        return OrderDetails(
            order_id="exit-1",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(order_manager, "_place_exit_order", _fake_exit_order)

    intent = ExitIntent(symbol="NIFTY24JUN25450CE", qty=225, product="MIS")
    order_id = order_manager.place_reduce_only_exit(intent)

    assert order_id == "exit-1"
    assert captured["quantity"] == 150
    assert captured["symbol"] == "NFO:NIFTY24JUN25450CE"
    assert captured["product"] == "MIS"
    assert captured["tag"] == "reduce_only"


def test_place_reduce_only_exit_no_position(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _StubBroker()
    broker.get_positions = MagicMock(return_value=[])
    position_manager = MagicMock()
    limiter = _StubRateLimiter()
    order_manager = OrderManager(broker, position_manager, limiter)

    def _fail_exit_order(
        **_kwargs: object,
    ) -> OrderDetails:  # pragma: no cover - defensive
        raise AssertionError("reduce-only guard should skip exit placement")

    monkeypatch.setattr(order_manager, "_place_exit_order", _fail_exit_order)

    intent = ExitIntent(symbol="NIFTY24JUN25450CE", qty=75, product="MIS")
    order_id = order_manager.place_reduce_only_exit(intent)

    assert order_id is None
