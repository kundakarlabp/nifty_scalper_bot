from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SimulatedOrder:
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: float | None = None
    trigger_price: float | None = None
    status: str = "NEW"
    filled_quantity: int = 0
    average_price: float = 0.0
    tag: str | None = None

    @property
    def remaining_quantity(self) -> int:
        return max(self.quantity - self.filled_quantity, 0)


@dataclass(frozen=True)
class SimulatedTrade:
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float


class SimulatedBroker:
    """Deterministic broker/matcher.

    Long protective stops trigger on bid or LTP <= stop.
    """

    def __init__(self, clock, recorder=None) -> None:
        self.clock = clock
        self.recorder = recorder
        self.orders: dict[str, SimulatedOrder] = {}
        self.trades: list[SimulatedTrade] = []
        self.positions: dict[str, int] = {}
        self.realised_pnl: dict[str, float] = {}
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._seq = 0
        self._trade_seq = 0
        self._entry_cost: dict[str, float] = {}
        self._seen: set[tuple[str, int, float]] = set()

    def register_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float | None = None,
        trigger_price: float | None = None,
        tag: str | None = None,
        reject_reason: str | None = None,
        delayed_ack_seconds: float = 0.0,
    ) -> str:
        self._seq += 1
        oid = f"SIM{self._seq:06d}"
        order = SimulatedOrder(
            oid, symbol, side, quantity, order_type, price, trigger_price, tag=tag
        )
        self.orders[oid] = order
        if reject_reason:
            order.status = "REJECTED"
            self._emit(order, reason=reject_reason)
            return oid

        def ack() -> None:
            order.status = "TRIGGER_PENDING" if order_type in {"SL", "SL-M"} else "OPEN"
            self._emit(order)

        (
            self.clock.call_later(delayed_ack_seconds, ack)
            if delayed_ack_seconds
            else ack()
        )
        return oid

    def modify_order(self, order_id: str, **changes: Any) -> None:
        order = self.orders[order_id]
        for key in ("quantity", "price", "trigger_price"):
            if key in changes and changes[key] is not None:
                setattr(order, key, changes[key])
        self._emit(order, modified=True)

    def cancel_order(self, order_id: str) -> None:
        order = self.orders[order_id]
        if order.status != "COMPLETE":
            order.status = "CANCELLED"
            self._emit(order)

    def query_order(self, order_id: str) -> SimulatedOrder:
        return self.orders[order_id]

    def query_orders(self) -> list[SimulatedOrder]:
        return list(self.orders.values())

    def query_trades(self) -> list[SimulatedTrade]:
        return list(self.trades)

    def query_positions(self) -> dict[str, int]:
        return dict(self.positions)

    def query_holdings(self) -> list[Any]:
        return []

    def on_quote(self, symbol: str, *, bid: float, ask: float, ltp: float) -> None:
        for order in list(self.orders.values()):
            if (
                order.symbol == symbol
                and order.remaining_quantity
                and order.status in {"OPEN", "TRIGGER_PENDING", "PARTIALLY_FILLED"}
                and self._should_fill(order, bid, ask, ltp)
            ):
                self.fill(
                    order.order_id,
                    order.remaining_quantity,
                    self._fill_price(order, bid, ask, ltp),
                )

    def fill(
        self, order_id: str, quantity: int, price: float, *, duplicate: bool = False
    ) -> None:
        order = self.orders[order_id]
        key = (order_id, quantity, price)
        if duplicate and key in self._seen:
            self._emit(order, duplicate=True)
            return
        self._seen.add(key)
        quantity = min(quantity, order.remaining_quantity)
        if quantity <= 0:
            return
        value = order.average_price * order.filled_quantity + quantity * price
        order.filled_quantity += quantity
        order.average_price = value / order.filled_quantity
        order.status = (
            "COMPLETE" if order.remaining_quantity == 0 else "PARTIALLY_FILLED"
        )
        self._trade_seq += 1
        self.trades.append(
            SimulatedTrade(
                f"T{self._trade_seq:06d}",
                order_id,
                order.symbol,
                order.side,
                quantity,
                price,
            )
        )
        signed = quantity if order.side == "BUY" else -quantity
        before = self.positions.get(order.symbol, 0)
        self.positions[order.symbol] = before + signed
        if order.side == "BUY":
            self._entry_cost[order.symbol] = (
                self._entry_cost.get(order.symbol, 0.0) + quantity * price
            )
        else:
            avg_cost = self._entry_cost.get(order.symbol, 0.0) / max(before, 1)
            self.realised_pnl[order.symbol] = self.realised_pnl.get(
                order.symbol, 0.0
            ) + quantity * (price - avg_cost)
            self._entry_cost[order.symbol] = max(
                self._entry_cost.get(order.symbol, 0.0) - quantity * avg_cost, 0.0
            )
        self._emit(order, fill_quantity=quantity, fill_price=price)

    def _should_fill(
        self, order: SimulatedOrder, bid: float, ask: float, ltp: float
    ) -> bool:
        if order.order_type == "MARKET":
            return True
        if order.side == "BUY" and order.order_type == "LIMIT":
            return ask <= float(order.price)
        if order.side == "SELL" and order.order_type == "LIMIT":
            return bid >= float(order.price)
        if order.side == "SELL" and order.order_type in {"SL", "SL-M"}:
            stop = float(
                order.trigger_price if order.trigger_price is not None else order.price
            )
            return bid <= stop or ltp <= stop
        return False

    def _fill_price(
        self, order: SimulatedOrder, bid: float, ask: float, ltp: float
    ) -> float:
        return (
            ask
            if order.side == "BUY" and order.order_type == "MARKET"
            else float(order.price or bid)
        )

    def _emit(self, order: SimulatedOrder, **payload: Any) -> None:
        update = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "status": order.status,
            "filled_quantity": order.filled_quantity,
            "average_price": order.average_price,
            **payload,
        }
        for cb in list(self._callbacks):
            cb(update)

    @property
    def pending_callbacks(self) -> int:
        return 0
