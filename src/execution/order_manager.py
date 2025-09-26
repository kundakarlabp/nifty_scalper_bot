"""Order management helpers specialised for the simplified scalper flow."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional


log = logging.getLogger(__name__)


FINAL_STATES = {"COMPLETE", "REJECTED", "CANCELLED"}
OPENISH_STATES = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}


StatusFetcher = Callable[[str], Mapping[str, Any] | str | None]
CancelFunc = Callable[[str], None]
PlaceFunc = Callable[[MutableMapping[str, Any]], Optional[str]]
SquareOffFunc = Callable[[str, str, Optional[int]], None]
OrdersFetcher = Callable[[], Iterable[Mapping[str, Any]]]


class OrderManager:
    """Place orders and wait for confirmations before proceeding."""

    def __init__(
        self,
        place_order: PlaceFunc,
        *,
        status_fetcher: StatusFetcher | None = None,
        cancel_order: CancelFunc | None = None,
        square_off: SquareOffFunc | None = None,
        poll_interval: float = 1.0,
        orders_fetcher: OrdersFetcher | None = None,
        kite: Any | None = None,
    ) -> None:
        self._place_order = place_order
        self._status_fetcher = status_fetcher
        self._cancel_order = cancel_order
        self._square_off = square_off
        self._poll_interval = max(0.0, float(poll_interval))
        self._orders_fetcher = orders_fetcher
        self._kite = kite
        # ``self.kite`` mirrors ``self._kite`` to match older integrations
        # that accessed the broker handle directly on :class:`OrderManager`.
        self.kite = kite

    # ------------------------------------------------------------------
    def place_order_with_confirmation(
        self, params: Mapping[str, Any], *, max_wait_sec: float = 10.0
    ) -> Optional[str]:
        """Place an order and wait for a ``COMPLETE`` confirmation."""

        payload: MutableMapping[str, Any] = dict(params)
        try:
            order_id = self._place_order(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("place_order failed: %s", exc)
            return None

        if not order_id:
            return None
        if not self._status_fetcher and not self._orders_fetcher and not self._kite:
            return order_id

        deadline = time.monotonic() + max(0.0, float(max_wait_sec))
        while time.monotonic() <= deadline:
            record = self._locate_order(order_id)
            status = self._extract_status(record) if record is not None else None
            if status == "COMPLETE":
                return order_id
            if status in {"REJECTED", "CANCELLED"}:
                return None
            time.sleep(self._poll_interval)

        self._cancel(order_id)
        return None

    # ------------------------------------------------------------------
    def square_off_position(
        self, symbol: str, *, side: str, quantity: int | None = None
    ) -> None:
        """Exit an existing position, falling back to a market order.

        Parameters
        ----------
        symbol:
            Tradingsymbol of the option/underlying to be squared off.
        side:
            The transaction side that should be used to exit the position
            (``"BUY"`` to close shorts, ``"SELL"`` to close longs).
        quantity:
            Optional quantity for the closing order.
        """

        exit_side = str(side).upper()

        if self._square_off is not None:
            try:
                self._square_off(symbol, exit_side, quantity)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("square_off callback failed for %s: %s", symbol, exc)
            return

        if not symbol:
            raise ValueError("symbol must be provided for square off")

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "transaction_type": exit_side,
            "order_type": "MARKET",
        }
        if quantity is not None:
            payload["quantity"] = int(quantity)
        try:
            self._place_order(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("square_off_position failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    def _get_order_status(self, order_id: str) -> str | None:
        record = self._locate_order(order_id)
        return self._extract_status(record)

    # ------------------------------------------------------------------
    def _cancel(self, order_id: str) -> None:
        self._cancel_with_variety(order_id, {})

    # ------------------------------------------------------------------
    def _locate_order(self, order_id: str) -> Mapping[str, Any] | None:
        """Return the full order record when available."""

        if self._orders_fetcher is not None:
            try:
                for order in self._orders_fetcher() or []:
                    if str(order.get("order_id")) == str(order_id):
                        return order
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("orders_fetcher failed: %s", exc)

        try:
            record = self._kite_orders(order_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("order lookup failed for %s: %s", order_id, exc)
        else:
            if record:
                return record

        try:
            record = self._status_fetcher_lookup(order_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("order lookup failed for %s: %s", order_id, exc)
        else:
            if record:
                return record
        return None

    def _kite_orders(self, order_id: str) -> Mapping[str, Any] | None:
        if not self._kite:
            return None
        orders_fn = getattr(self._kite, "orders", None)
        if not callable(orders_fn):
            return None
        try:  # pragma: no cover - network
            orders = orders_fn()
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("kite.orders failed: %s", exc)
            return None
        for order in orders or []:
            if str(order.get("order_id")) == str(order_id):
                return order
        return None

    def _status_fetcher_lookup(self, order_id: str) -> Mapping[str, Any] | None:
        if not self._status_fetcher:
            return None
        result = self._status_fetcher(order_id)
        if isinstance(result, Mapping):
            return result
        if result is None:
            return None
        return {"status": result}

    def _extract_status(self, record: Mapping[str, Any] | None) -> str | None:
        if not isinstance(record, Mapping):
            return None
        for key in ("status", "order_status", "state"):
            val = record.get(key)
            if val:
                status = str(val).upper()
                return status or None
        return None

    # ------------------------------------------------------------------
    def _cancel_with_variety(self, order_id: str, last: Mapping[str, Any]) -> None:
        variety = None
        if isinstance(last, Mapping):
            variety = last.get("variety") or last.get("order_variety")
        variety = variety or "regular"

        if self._kite:
            cancel_fn = getattr(self._kite, "cancel_order", None)
            if callable(cancel_fn):
                try:  # pragma: no cover - network
                    cancel_fn(variety=variety, order_id=order_id)
                    return
                except Exception as exc:  # pragma: no cover - defensive logging
                    log.warning("cancel_failed order_id=%s: %s", order_id, exc)

        if self._cancel_order:
            try:
                self._cancel_order(order_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("cancel_order failed for %s: %s", order_id, exc)

    # ------------------------------------------------------------------
    def _confirm_order(
        self, order_id: str | None, timeout: int = 15
    ) -> Mapping[str, Any]:
        """Poll order status until completion or timeout then cancel."""

        if not order_id:
            return {"order_id": order_id, "status": "ERROR"}

        deadline = time.monotonic() + max(timeout, 0)
        poll_interval = max(self._poll_interval, 0.5)
        last: Dict[str, Any] = {"order_id": order_id}

        while time.monotonic() < deadline:
            try:
                record = self._locate_order(order_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("order status check failed: %s", exc)
                record = None

            if record:
                last = dict(record)
                status = self._extract_status(record)
                if status in FINAL_STATES:
                    last.setdefault("status", status)
                    return self._finalise_order_record(order_id, last)

            time.sleep(poll_interval)

        status = self._extract_status(last)
        if status and status not in OPENISH_STATES:
            last.setdefault("status", status)
            return self._finalise_order_record(order_id, last)

        self._cancel_with_variety(order_id, last)
        last.setdefault("status", "TIMEOUT")
        return self._finalise_order_record(order_id, last)

    # ------------------------------------------------------------------
    def _finalise_order_record(
        self, order_id: str, snapshot: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        """Merge snapshot with order history and log explicit outcomes."""

        merged: Dict[str, Any] = dict(snapshot or {})
        merged.setdefault("order_id", order_id)

        history_entry: Dict[str, Any] | None = None
        history_fn = getattr(self._kite, "order_history", None) if self._kite else None
        if callable(history_fn):
            try:  # pragma: no cover - network defensive guard
                history = history_fn(order_id) or []
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning("order_history failed for %s: %s", order_id, exc)
            else:
                if history:
                    history_entry = dict(history[-1])

        if history_entry:
            merged.update(history_entry)
            merged.setdefault("order_id", order_id)

        status = str(merged.get("status") or "").upper()
        if status == "REJECTED":
            reason = (
                history_entry.get("rejection_reason")
                if history_entry
                else merged.get("rejection_reason")
            ) or merged.get("message") or "unknown"
            log.error("Order REJECTED id=%s reason=%s", order_id, reason)
        elif status == "CANCELLED":
            log.warning("Order CANCELLED id=%s", order_id)
        elif status == "COMPLETE":
            log.info(
                "Order COMPLETE id=%s avg_price=%s",
                order_id,
                merged.get("average_price"),
            )

        return merged or {"order_id": order_id, "status": "UNKNOWN"}

    # ------------------------------------------------------------------
    def place_straddle_orders(
        self,
        ce_params: Mapping[str, Any],
        pe_params: Mapping[str, Any],
        *,
        confirm_timeout: int = 15,
    ) -> bool:
        """Place CE/PE legs, confirm fills, and handle partial executions."""

        ce_id = self.place_order_with_confirmation(ce_params, max_wait_sec=0.0)
        pe_id = self.place_order_with_confirmation(pe_params, max_wait_sec=0.0)

        ce_info = self._confirm_order(ce_id, confirm_timeout) if ce_id else {"status": "ERROR"}
        pe_info = self._confirm_order(pe_id, confirm_timeout) if pe_id else {"status": "ERROR"}

        ce_status = self._extract_status(ce_info) or str(ce_info.get("status", "")).upper()
        pe_status = self._extract_status(pe_info) or str(pe_info.get("status", "")).upper()

        ce_complete = ce_status == "COMPLETE"
        pe_complete = pe_status == "COMPLETE"
        if ce_complete and pe_complete:
            return True

        ce_symbol = self._resolve_symbol(ce_params)
        pe_symbol = self._resolve_symbol(pe_params)
        ce_side = self._resolve_side(ce_params)
        pe_side = self._resolve_side(pe_params)
        qty_ce = self._resolve_quantity(ce_params)
        qty_pe = self._resolve_quantity(pe_params)

        if ce_complete and not pe_complete and ce_symbol:
            self.square_off_position(ce_symbol, side=self._opposite_side(ce_side), quantity=qty_ce)
            log.warning("PE leg failed; squared off CE leg for %s", ce_symbol)
        elif pe_complete and not ce_complete and pe_symbol:
            self.square_off_position(pe_symbol, side=self._opposite_side(pe_side), quantity=qty_pe)
            log.warning("CE leg failed; squared off PE leg for %s", pe_symbol)

        return ce_complete and pe_complete

    # ------------------------------------------------------------------
    def _resolve_symbol(self, params: Mapping[str, Any]) -> str | None:
        for key in ("symbol", "tradingsymbol"):
            symbol = params.get(key)
            if symbol:
                return str(symbol)
        return None

    def _resolve_side(self, params: Mapping[str, Any]) -> str:
        side = params.get("transaction_type") or params.get("side") or "BUY"
        return str(side).upper()

    def _resolve_quantity(self, params: Mapping[str, Any]) -> int | None:
        qty = params.get("quantity") or params.get("qty")
        if qty is None:
            return None
        try:
            return int(qty)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _opposite_side(self, side: str) -> str:
        return "SELL" if str(side).upper() == "BUY" else "BUY"


__all__ = ["OrderManager"]

