"""Kite broker adapter wrapping `kiteconnect` SDK."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

try:  # pragma: no cover - optional dependency
    import kiteconnect.exceptions as kex  # type: ignore
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    kex = None  # type: ignore

from src.broker.instruments import InstrumentStore
from src.broker.interface import (
    AuthError,
    Broker,
    BrokerError,
    InsufficientMargin,
    NetworkError,
    Order,
    OrderNotFound,
    OrderRejected,
    OrderRequest,
    OrderStatus,
    OrderType,
    RateLimitError,
    Side,
    Tick,
    TimeInForce,
)
from src.execution import broker_retry
from src.utils.broker_errors import (
    AUTH,
    THROTTLE,
    classify_broker_error,
)
from src.utils.broker_errors import (
    NETWORK as NET,
)

logger = logging.getLogger(__name__)


class KiteBroker(Broker):
    """Broker implementation for Zerodha Kite (simplified)."""

    def __init__(
        self,
        api_key: str,
        access_token: str,
        instrument_store: InstrumentStore | None = None,
        enable_ws: bool = True,
    ) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.enable_ws = enable_ws
        self.instrument_store = instrument_store or InstrumentStore()
        self._kite: Any = None
        self._connected = False
        self._tick_cb: Callable[[Tick], None] | None = None

    # Connection and market data -------------------------------------------------
    def connect(self) -> None:
        if KiteConnect is None:
            raise BrokerError("kiteconnect package not installed")
        try:
            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)
            self._connected = True
            logger.info("Kite REST connected")
        except Exception as exc:
            logger.exception("Kite connect failed")
            self._translate_and_raise(exc)

    def is_connected(self) -> bool:
        return bool(self._connected)

    def subscribe_ticks(
        self, instruments: Sequence[int], on_tick: Callable[[Tick], None]
    ) -> None:  # pragma: no cover - WS omitted
        self._tick_cb = on_tick

    def ltp(self, instrument_id: int) -> Decimal:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        data = broker_retry.call(self._kite.ltp, [instrument_id])  # type: ignore[operator]
        first = next(iter(data.values()))
        return Decimal(str(first.get("last_price")))

    # Order management ----------------------------------------------------------
    def place_order(
        self, req: OrderRequest
    ) -> str:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        params = self._map_order_request(req)
        oid = self._kite.place_order(**params)  # type: ignore[call-arg]
        return str(oid)

    def modify_order(
        self, order_id: str, **kwargs: Any
    ) -> None:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        self._kite.modify_order(order_id=order_id, **kwargs)  # type: ignore[call-arg]

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        self._kite.cancel_order(order_id=order_id)  # type: ignore[call-arg]

    def get_order(self, order_id: str) -> Order:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        hist = broker_retry.call(self._kite.order_history, order_id=order_id)  # type: ignore[call-arg]
        payload = hist[-1] if hist else {"order_id": order_id, "status": "UNKNOWN"}
        return self._map_order(payload)

    def get_positions(
        self,
    ) -> list[dict[str, Any]]:  # pragma: no cover - network omitted
        if self._kite is None:
            raise BrokerError("Kite not connected")
        pos = broker_retry.call(self._kite.positions)  # type: ignore[call-arg]
        return pos.get("net", []) if isinstance(pos, dict) else pos

    # Mapping helpers -----------------------------------------------------------
    def _map_order_request(self, req: OrderRequest) -> dict[str, Any]:
        inst = self.instrument_store.by_token(req.instrument_id)
        side = "BUY" if req.side == Side.BUY else "SELL"
        variety = inst.variety if inst else "regular"
        product = inst.product if inst else "MIS"
        exchange = inst.exchange if inst else "NFO"
        tradingsymbol = inst.symbol if inst else str(req.instrument_id)

        params: dict[str, Any] = dict(
            variety=variety,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            transaction_type=side,
            quantity=int(req.qty),
            product=product,
            order_type="MARKET" if req.order_type == OrderType.MARKET else "LIMIT",
        )
        if req.order_type == OrderType.LIMIT and req.price is not None:
            params["price"] = float(req.price)
        if req.client_order_id:
            params["tag"] = req.client_order_id
        if req.tif == TimeInForce.IOC:
            params["validity"] = "IOC"
        elif req.tif == TimeInForce.GTC:
            params["validity"] = "GTC"
        else:
            params["validity"] = "DAY"
        return params

    def _map_order(self, payload: dict[str, Any]) -> Order:
        status_map = {
            "OPEN": OrderStatus.OPEN,
            "TRIGGER PENDING": OrderStatus.OPEN,
            "PARTIALLY FILLED": OrderStatus.PARTIALLY_FILLED,
            "COMPLETE": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
        }
        status = status_map.get(
            str(payload.get("status", "")).upper(), OrderStatus.OPEN
        )
        filled = int(payload.get("filled_quantity", payload.get("filled_qty", 0)) or 0)
        avg = payload.get("average_price") or payload.get("avg_price")
        avg_dec = Decimal(str(avg)) if avg is not None else None
        return Order(
            order_id=str(payload.get("order_id", "")),
            status=status,
            filled_qty=filled,
            avg_fill_price=avg_dec,
            raw=payload,
        )

    # Error translation ---------------------------------------------------------
    def _translate_and_raise(self, exc: Exception) -> None:
        if kex is not None:
            if isinstance(exc, getattr(kex, "TokenException", ())):
                raise AuthError(str(exc)) from exc
            if isinstance(exc, getattr(kex, "NetworkException", ())):
                raise NetworkError(str(exc)) from exc
            if isinstance(exc, getattr(kex, "RateLimitException", ())):
                raise RateLimitError(str(exc)) from exc
            if isinstance(exc, getattr(kex, "PermissionException", ())):
                raise AuthError(str(exc)) from exc
            if isinstance(exc, getattr(kex, "InputException", ())):
                raise OrderRejected(str(exc)) from exc
            if isinstance(exc, getattr(kex, "OrderException", ())):
                msg = str(exc)
                if "margin" in msg.lower():
                    raise InsufficientMargin(msg) from exc
                raise OrderRejected(msg) from exc
            if isinstance(exc, getattr(kex, "GeneralException", ())):
                raise BrokerError(str(exc)) from exc
        msg = str(exc)
        kind = classify_broker_error(exc)
        if kind == AUTH:
            raise AuthError(msg) from exc
        if kind == THROTTLE:
            raise RateLimitError(msg) from exc
        if kind == NET:
            raise NetworkError(msg) from exc
        if "margin" in msg.lower():
            raise InsufficientMargin(msg) from exc
        if "reject" in msg.lower():
            raise OrderRejected(msg) from exc
        if "not found" in msg.lower():
            raise OrderNotFound(msg) from exc
        raise BrokerError(msg) from exc
