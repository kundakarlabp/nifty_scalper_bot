"""In-memory broker used for tests."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.broker.interface import Broker, Order, OrderRequest, OrderStatus, Tick


class MockBroker(Broker):
    """Simple broker that immediately fills orders and publishes ticks."""

    def __init__(self) -> None:
        self._connected = False
        self._tick_cb: Optional[Callable[[Tick], None]] = None
        self._orders: Dict[str, Order] = {}
        self._oid = 0

    def connect(self) -> None:  # noqa: D401 - brief override
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def subscribe_ticks(self, instruments: Sequence[int], on_tick: Callable[[Tick], None]) -> None:
        self._tick_cb = on_tick

    def push_tick(self, instrument_id: int, ltp: Decimal) -> None:
        if self._tick_cb:
            self._tick_cb(Tick(instrument_id=instrument_id, ts=time.time(), ltp=ltp))

    def ltp(self, instrument_id: int) -> Decimal:  # pragma: no cover - trivial
        return Decimal("100.00")

    def place_order(self, req: OrderRequest) -> str:
        self._oid += 1
        oid = f"MOCK-{self._oid}"
        order = Order(order_id=oid, status=OrderStatus.FILLED, filled_qty=req.qty, avg_fill_price=req.price)
        self._orders[oid] = order
        return oid

    def modify_order(self, order_id: str, **kwargs: Any) -> None:  # pragma: no cover - unused
        pass

    def cancel_order(self, order_id: str) -> None:
        if order_id in self._orders:
            existing = self._orders[order_id]
            self._orders[order_id] = Order(
                order_id=existing.order_id,
                status=OrderStatus.CANCELLED,
                filled_qty=existing.filled_qty,
                avg_fill_price=existing.avg_fill_price,
            )

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def get_positions(self) -> List[Dict[str, Any]]:  # pragma: no cover - unused
        return []
