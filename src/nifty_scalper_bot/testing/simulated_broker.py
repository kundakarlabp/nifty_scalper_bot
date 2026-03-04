"""Simulated Zerodha-compatible in-memory broker."""

from __future__ import annotations

import random
import uuid
from typing import Any


class SimulatedZerodhaBroker:
    """Minimal broker simulator for replay/paper testing."""

    def __init__(self) -> None:
        """Args: none; Returns: None; Raises: None."""
        self.orders: dict[str, dict[str, Any]] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.price_feed: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Args: symbol, price; Returns: None; Raises: None."""
        self.price_feed[symbol] = float(price)

    def get_ltp(self, symbol: str) -> float | None:
        """Args: symbol; Returns: float|None; Raises: None."""
        return self.price_feed.get(symbol)

    def place_order(self, **kwargs: Any) -> dict[str, str]:
        """Args: kwargs; Returns: dict; Raises: ValueError."""
        symbol = str(kwargs.get("symbol") or "")
        side = str(kwargs.get("side") or kwargs.get("transaction_type") or "BUY")
        quantity = int(kwargs.get("quantity") or kwargs.get("qty") or 0)
        order_type = str(kwargs.get("order_type") or kwargs.get("type") or "MARKET")
        price = kwargs.get("price")
        if not symbol or quantity <= 0:
            raise ValueError("invalid order payload")
        oid = str(uuid.uuid4())
        if order_type.upper() == "MARKET":
            price = self.get_ltp(symbol)
        ltp = float(price or 0.0)
        fill_price = ltp * (1 + random.uniform(-0.002, 0.002))
        if random.random() < 0.3:
            filled_qty = max(1, int(quantity * 0.5))
            status = "PARTIAL"
        else:
            filled_qty = quantity
            status = "FILLED"
        self.orders[oid] = {
            "order_id": oid,
            "symbol": symbol,
            "side": side.upper(),
            "qty": quantity,
            "filled_qty": filled_qty,
            "price": fill_price,
            "status": status,
        }
        self._update_position(symbol, side, filled_qty, fill_price)
        return {"order_id": oid}

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        """Args: order_id; Returns: bool; Raises: None."""
        _ = args, kwargs
        if order_id not in self.orders:
            return False
        self.orders[order_id]["status"] = "CANCELLED"
        return True

    def get_positions(self) -> list[dict[str, Any]]:
        """Args: none; Returns: list; Raises: None."""
        return [dict(v) for v in self.positions.values()]

    def get_orders(self) -> list[dict[str, Any]]:
        """Args: none; Returns: list; Raises: None."""
        return [dict(v) for v in self.orders.values()]

    def _update_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> None:
        """Args: symbol, side, quantity, price; Returns: None; Raises: None."""
        record = self.positions.setdefault(
            symbol,
            {
                "symbol": symbol,
                "qty": 0,
                "avg_price": 0.0,
            },
        )
        signed_qty = quantity if side.upper() == "BUY" else -quantity
        new_qty = int(record["qty"]) + signed_qty
        if new_qty == 0:
            record["qty"] = 0
            record["avg_price"] = 0.0
            return
        record["qty"] = new_qty
        record["avg_price"] = float(price)
