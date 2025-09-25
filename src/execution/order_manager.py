"""Order management helpers specialised for the simplified scalper flow."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional


log = logging.getLogger(__name__)


StatusFetcher = Callable[[str], Mapping[str, Any] | str | None]
CancelFunc = Callable[[str], None]
PlaceFunc = Callable[[MutableMapping[str, Any]], Optional[str]]
SquareOffFunc = Callable[[str, str, Optional[int]], None]


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
    ) -> None:
        self._place_order = place_order
        self._status_fetcher = status_fetcher
        self._cancel_order = cancel_order
        self._square_off = square_off
        self._poll_interval = max(0.0, float(poll_interval))

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
        if not self._status_fetcher:
            return order_id

        deadline = time.monotonic() + max(0.0, float(max_wait_sec))
        while time.monotonic() <= deadline:
            status = self._get_order_status(order_id)
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
        """Exit an existing position, falling back to a market order."""

        if self._square_off is not None:
            try:
                self._square_off(symbol, side, quantity)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("square_off callback failed for %s: %s", symbol, exc)
            return

        if not symbol:
            raise ValueError("symbol must be provided for square off")

        opposite = "SELL" if side.upper() == "BUY" else "BUY"
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "transaction_type": opposite,
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
        if not self._status_fetcher:
            return None
        try:
            status_raw = self._status_fetcher(order_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("status fetch failed for %s: %s", order_id, exc)
            return None

        if isinstance(status_raw, Mapping):
            for key in ("status", "order_status", "state"):
                candidate = status_raw.get(key)
                if candidate:
                    status_raw = candidate
                    break

        if status_raw is None:
            return None
        status = str(status_raw).upper()
        return status or None

    # ------------------------------------------------------------------
    def _cancel(self, order_id: str) -> None:
        if not self._cancel_order:
            return
        try:
            self._cancel_order(order_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("cancel_order failed for %s: %s", order_id, exc)


__all__ = ["OrderManager"]

