"""Enhanced paper trading fill engine with queue-aware fills."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, Mapping, cast

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return default
        try:
            return float(token)
        except ValueError:
            return default
    return default


@dataclass(slots=True)
class _QuoteMetrics:
    ltp: float
    bid: float
    ask: float
    mid: float
    spread: float
    momentum: float


class PaperFillEngine:
    """Simulate broker order placement using cached market data."""

    def __init__(self, data_hub: DataHub, resolver: Any) -> None:
        self.hub = data_hub
        self.resolver = resolver
        self.orders: Dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._clock = time.time
        self._children: dict[str, list[str]] = {}
        self._queue_depth = self._coerce_int(os.getenv("PAPER__QUEUE_DEPTH", "0"))
        self._respect_lot = self._coerce_bool(
            os.getenv("PAPER__RESPECT_LOTSIZE", "true")
        )
        self._slip_bps = max(0.0, _safe_float(os.getenv("PAPER__SLIPPAGE_BPS"), 0.0))
        self._slippage_model = SlippageModel()

    # ------------------------------------------------------------------
    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Simulate order placement returning a broker-like response."""

        symbol = DataHub.normalize(
            payload.get("tradingsymbol") or payload.get("symbol") or ""
        )
        side = str(payload.get("transaction_type") or payload.get("side") or "").upper()
        quantity = self._coerce_int(payload.get("quantity"), default=0)
        if quantity <= 0:
            return {
                "order_id": "",
                "status": "rejected",
                "message": "Quantity must be positive",
            }
        if self._respect_lot and symbol.endswith(("CE", "PE")):
            lot = self._resolve_lot(symbol)
            if quantity % lot != 0:
                return {
                    "order_id": "",
                    "status": "rejected",
                    "message": f"Qty must be multiple of {lot}",
                }

        raw_quote = self.hub.get_quote(symbol, allow_pull=True)
        quote: Mapping[str, Any] = raw_quote or {}
        metrics = self._quote_metrics(quote)

        order_type = str(payload.get("order_type") or "MARKET").upper()
        limit_price = payload.get("price")

        order_id = self._next_id()
        parent_id_raw = payload.get("parent_order_id")
        parent_id = None
        if isinstance(parent_id_raw, str):
            parent_id = parent_id_raw.strip() or None
        elif parent_id_raw is not None:
            parent_id = str(parent_id_raw)

        order = {
            "order_id": order_id,
            "status": "open",
            "filled_quantity": 0,
            "average_price": None,
            "message": None,
            "quantity": quantity,
            "symbol": symbol,
            "transaction_type": side,
            "order_type": order_type,
            "price": _safe_float(limit_price, 0.0) if limit_price is not None else None,
            "timestamp": self._clock(),
            "parent_order_id": parent_id,
            "remaining_quantity": quantity,
        }

        if parent_id:
            self._children.setdefault(parent_id, []).append(order_id)

        self._apply_fill(order, metrics)
        self.orders[order_id] = order
        return dict(order)

    def modify_order(
        self,
        order_id: str,
        *,
        quantity: int | None = None,
        price: float | None = None,
        variety: str | None = None,
    ) -> dict[str, Any]:
        """Emulate order modification by updating stored payload."""

        order = self.orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order_id {order_id}")
        order = cast(dict[str, Any], order)
        if quantity is not None:
            order["quantity"] = max(0, int(quantity))
        if price is not None:
            order["price"] = float(price)
        order["timestamp"] = self._clock()
        order.setdefault("filled_quantity", 0)
        order.setdefault("remaining_quantity", order.get("quantity", 0))
        order_type = str(order.get("order_type", "")).upper()
        if order_type != "MARKET":
            metrics = self._quote_metrics(
                self.hub.get_quote(order["symbol"], allow_pull=True) or {}
            )
            self._apply_fill(order, metrics)
        return dict(order)

    def cancel_order(self, order_id: str, variety: str | None = None) -> dict[str, Any]:
        """Mark order as cancelled, cascading to bracket children."""

        order = self.orders.get(order_id)
        if order is None:
            return {"order_id": order_id, "status": "rejected", "message": "not_found"}
        order["status"] = "cancelled"
        order["timestamp"] = self._clock()
        order["remaining_quantity"] = max(
            0, int(order.get("quantity", 0)) - int(order.get("filled_quantity", 0))
        )
        for child_id in self._children.get(order_id, []):
            child = self.orders.get(child_id)
            if child is None:
                continue
            child["status"] = "cancelled"
            child["timestamp"] = self._clock()
            child["message"] = "Parent cancelled"
        return dict(order)

    def get_orders(self) -> list[dict[str, Any]]:
        """Return tracked simulated orders."""

        return [dict(order) for order in self.orders.values()]

    # ------------------------------------------------------------------
    def _apply_fill(self, order: dict[str, Any], metrics: _QuoteMetrics) -> None:
        side = str(order.get("transaction_type") or "").upper()
        order_type = str(order.get("order_type") or "").upper()
        quantity = int(order.get("quantity") or 0)
        if quantity <= 0:
            order.update({"status": "rejected", "message": "Quantity must be positive"})
            return

        if order_type == "MARKET":
            self._fill_market(order, metrics)
            return

        limit_price = order.get("price")
        if limit_price is None:
            order["status"] = "open"
            order["remaining_quantity"] = quantity - int(
                order.get("filled_quantity", 0)
            )
            return

        ref_price = metrics.ask if side == "BUY" else metrics.bid
        if ref_price <= 0:
            ref_price = metrics.mid or metrics.ltp
        fillable = False
        if side == "BUY":
            fillable = limit_price >= ref_price
        else:
            fillable = limit_price <= ref_price

        if not fillable:
            order["status"] = "open"
            order["remaining_quantity"] = quantity - int(
                order.get("filled_quantity", 0)
            )
            return

        partial = self._consume_queue(quantity)
        if partial <= 0:
            order["status"] = "open"
            order["remaining_quantity"] = quantity - int(
                order.get("filled_quantity", 0)
            )
            return

        previous_fill = int(order.get("filled_quantity", 0))
        new_total = min(quantity, previous_fill + partial)
        weighted_price = limit_price
        if previous_fill > 0 and order.get("average_price") is not None:
            weighted_price = (
                previous_fill * float(order["average_price"])
                + partial * float(limit_price)
            ) / new_total

        order["filled_quantity"] = new_total
        order["average_price"] = weighted_price
        order["remaining_quantity"] = max(quantity - new_total, 0)
        order["status"] = "complete" if new_total >= quantity else "open"

    def _fill_market(self, order: dict[str, Any], metrics: _QuoteMetrics) -> None:
        side = str(order.get("transaction_type") or "").upper()
        quantity = int(order.get("quantity") or 0)
        base_price = metrics.mid or metrics.ltp
        if base_price <= 0:
            base_price = metrics.bid if side == "BUY" else metrics.ask
        spread_component = metrics.spread * (1.0 + abs(metrics.momentum))
        slip_component = self._slip_bps / 10000.0 * base_price
        model_adjustment = self._slippage_model.estimate(
            symbol=str(order.get("symbol") or ""),
            side=side,
            quantity=quantity,
            base_price=base_price,
        )
        total_adjustment = max(spread_component, slip_component) + abs(model_adjustment)
        LOGGER.debug(
            "paper_fill_market_slippage",
            extra={
                "event": "paper_fill_market_slippage",
                "symbol": order.get("symbol"),
                "side": side,
                "quantity": quantity,
                "base_price": base_price,
                "adjustment": total_adjustment,
            },
        )
        if side == "BUY":
            fill_price = base_price + total_adjustment
        else:
            fill_price = max(0.05, base_price - total_adjustment)
        order["filled_quantity"] = quantity
        order["average_price"] = fill_price
        order["remaining_quantity"] = 0
        order["status"] = "complete"

    # Helpers ------------------------------------------------------------
    def _next_id(self) -> str:
        self._sequence += 1
        return f"paper_{self._sequence}"

    @staticmethod
    def _coerce_int(raw: object, default: int = 0) -> int:
        value: int | None = None
        if isinstance(raw, bool):
            value = int(raw)
        elif isinstance(raw, int):
            value = raw
        elif isinstance(raw, float):
            value = int(raw)
        elif isinstance(raw, str):
            token = raw.strip()
            if not token:
                return default
            try:
                value = int(float(token))
            except ValueError:
                return default
        if value is None:
            return default
        return max(0, value)

    @staticmethod
    def _coerce_bool(raw: object, default: bool = True) -> bool:
        if raw is None:
            return default
        token = str(raw).strip().lower()
        if not token:
            return default
        if token in {"0", "false", "off", "no"}:
            return False
        if token in {"1", "true", "on", "yes"}:
            return True
        return default

    def _resolve_lot(self, symbol: str) -> int:
        if hasattr(self.resolver, "lot_size_for_symbol"):
            try:
                lot = int(self.resolver.lot_size_for_symbol(symbol))
                return max(lot, 1)
            except Exception:
                return 75
        return 75

    def _consume_queue(self, quantity: int) -> int:
        depth = max(self._queue_depth, 0)
        if depth <= 0:
            return quantity
        if quantity <= depth:
            self._queue_depth = depth - quantity
            return 0
        filled = quantity - depth
        self._queue_depth = 0
        return filled

    def _quote_metrics(self, quote: Mapping[str, Any]) -> _QuoteMetrics:
        bid = _safe_float(
            quote.get("best_bid")
            or quote.get("best_bid_price")
            or quote.get("bid")
            or quote.get("buy_price"),
            0.0,
        )
        ask = _safe_float(
            quote.get("best_ask")
            or quote.get("best_ask_price")
            or quote.get("ask")
            or quote.get("sell_price"),
            0.0,
        )
        ltp = _safe_float(quote.get("ltp") or quote.get("last_price"), 0.0)
        mid = 0.0
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif ltp > 0:
            mid = ltp
        spread = max(ask - bid, 0.0)
        if spread <= 0 and mid > 0:
            spread = mid * 0.0005
        close = _safe_float(quote.get("close") or quote.get("prev_close"), ltp)
        momentum = 0.0
        if close > 0:
            momentum = (ltp - close) / close
        return _QuoteMetrics(
            ltp=ltp, bid=bid, ask=ask, mid=mid, spread=spread, momentum=momentum
        )


__all__ = ["PaperFillEngine", "SlippageModel"]


@dataclass(slots=True)
class SlippageModel:
    """Estimate slippage adjustments for simulated fills."""

    base_tick: float = 0.05

    def estimate(
        self,
        symbol: str,
        side: str,
        quantity: int,
        base_price: float,
    ) -> float:
        """Return signed slippage adjustment for the provided order.

        Args:
            symbol: Instrument identifier used for heuristics.
            side: Trade direction ``"BUY"`` or ``"SELL"``.
            quantity: Requested order quantity.
            base_price: Reference price used for slippage computation.

        Returns:
            float: Signed price adjustment where positive increases price.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered SlippageModel.estimate",
            extra={
                "event": "paper_slippage_estimate",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        try:
            tick_size = self.base_tick
            normalized_symbol = str(symbol or "").upper()
            if "BANKNIFTY" in normalized_symbol or "FINNIFTY" in normalized_symbol:
                tick_size = 0.05
            elif "NIFTY" in normalized_symbol:
                tick_size = 0.05
            else:
                tick_size = 0.01
            quantity_factor = 1.0
            if quantity > 50:
                quantity_factor += 0.5
            if quantity > 100:
                quantity_factor += 0.5
            price_factor = 1.0
            if base_price >= 200.0:
                price_factor += 0.5
            adjustment = tick_size * quantity_factor * price_factor
            if side.upper() == "SELL":
                return -adjustment
            return adjustment
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in SlippageModel.estimate: %s",
                exc,
                extra={"event": "paper_slippage_estimate_error", "symbol": symbol},
                exc_info=exc,
            )
            return 0.0
