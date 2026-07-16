from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalize_side(value: Any) -> str:
    token = str(_enum_value(value) or "BUY").upper()
    if token.endswith(".BUY"):
        return "BUY"
    if token.endswith(".SELL"):
        return "SELL"
    if token in {"TRANSACTION_TYPE.BUY", "B"}:
        return "BUY"
    if token in {"TRANSACTION_TYPE.SELL", "S"}:
        return "SELL"
    return token


def _normalize_order_type(value: Any) -> str:
    token = str(_enum_value(value) or "MARKET").upper().replace(" ", "_")
    if token.endswith(".LIMIT") or token == "LIMIT":
        return "LIMIT"
    if token.endswith(".MARKET") or token == "MARKET":
        return "MARKET"
    if "STOP_LOSS_MARKET" in token or token in {"SL-M", "SLM", "STOPLOSSMARKET"}:
        return "SL-M"
    if "STOP_LOSS" in token or token in {"SL", "STOPLOSS"}:
        return "SL"
    return token


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
    intent: str | None = None

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
        self._fill_policies: dict[str, list[int]] = {}
        self._last_quotes: dict[str, tuple[float, float, float]] = {}

    def set_fill_policy(self, symbol: str, quantities: list[int]) -> None:
        self._fill_policies[symbol] = list(quantities)

    def register_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def place_order(
        self,
        *,
        symbol: str | None = None,
        side: str | None = None,
        quantity: int = 0,
        order_type: str = "MARKET",
        price: float | None = None,
        trigger_price: float | None = None,
        tag: str | None = None,
        reject_reason: str | None = None,
        delayed_ack_seconds: float = 0.0,
        **extra: Any,
    ) -> dict[str, Any]:
        symbol = str(
            symbol or extra.get("tradingsymbol") or extra.get("trading_symbol")
        )
        side = _normalize_side(
            side or extra.get("transaction_type") or extra.get("action") or "BUY"
        )
        order_type = _normalize_order_type(
            order_type or extra.get("variety") or "MARKET"
        )
        self._seq += 1
        oid = f"SIM{self._seq:06d}"
        intent = (
            str(
                extra.get("intent")
                or ("ENTRY" if str(tag or "").startswith("virtual_bra") else "")
            )
            or None
        )
        order = SimulatedOrder(
            oid,
            symbol,
            side,
            quantity,
            order_type,
            price,
            trigger_price,
            tag=tag,
            intent=intent,
        )
        self.orders[oid] = order
        if reject_reason:
            order.status = "REJECTED"
            self._emit(order, reason=reject_reason)
            return self._response(order, reason=reject_reason)

        def ack() -> None:
            order.status = "TRIGGER_PENDING" if order_type in {"SL", "SL-M"} else "OPEN"
            self._emit(order)

        (
            self.clock.call_later(delayed_ack_seconds, ack)
            if delayed_ack_seconds
            else ack()
        )
        return self._response(order)

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
        self._last_quotes[symbol] = (bid, ask, ltp)
        for order in list(self.orders.values()):
            if (
                order.symbol == symbol
                and order.remaining_quantity
                and order.status in {"OPEN", "TRIGGER_PENDING", "PARTIALLY_FILLED"}
                and self._should_fill(order, bid, ask, ltp)
            ):
                next_qty = order.remaining_quantity
                policy = self._fill_policies.get(order.symbol)
                if policy:
                    next_qty = min(policy.pop(0), order.remaining_quantity)
                self.fill(
                    order.order_id,
                    next_qty,
                    self._fill_price(order, bid, ask, ltp),
                )
                if policy and order.remaining_quantity:
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
            "tradingsymbol": order.symbol,
            "transaction_type": order.side,
            "status": order.status,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "pending_quantity": order.remaining_quantity,
            "average_price": order.average_price,
            "price": order.price,
            "trigger_price": order.trigger_price,
            "order_type": order.order_type,
            "tag": order.tag,
            "intent": order.intent,
            **payload,
        }
        for cb in list(self._callbacks):
            cb(update)

    def _response(self, order: SimulatedOrder, **payload: Any) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "status": order.status,
            "symbol": order.symbol,
            "tradingsymbol": order.symbol,
            "transaction_type": order.side,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "average_price": order.average_price,
            "price": order.price,
            "trigger_price": order.trigger_price,
            "order_type": order.order_type,
            "tag": order.tag,
            "intent": order.intent,
            **payload,
        }

    @property
    def pending_callbacks(self) -> int:
        return 0
