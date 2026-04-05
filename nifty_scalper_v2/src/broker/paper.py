"""Paper trading broker — simulated fills at mid-price."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .base import BrokerGateway


class PaperBroker(BrokerGateway):
    """Simulates order fills at mid-price with configurable slippage."""

    def __init__(self, slippage_model: str = "SPREAD_HALF") -> None:
        self._slippage_model = slippage_model
        self._orders: list[dict] = []
        self._positions: dict[str, dict] = {}
        self._last_quotes: dict[str, dict] = {}

    def update_quote(self, symbol: str, ltp: float, bid: float = 0.0, ask: float = 0.0) -> None:
        self._last_quotes[symbol] = {"ltp": ltp, "bid": bid, "ask": ask}

    def _fill_price(self, symbol: str, side: str) -> float:
        q = self._last_quotes.get(symbol, {})
        ltp = q.get("ltp", 0.0)
        bid = q.get("bid", 0.0)
        ask = q.get("ask", 0.0)

        if self._slippage_model == "ZERO":
            return ltp
        elif self._slippage_model == "SPREAD_HALF" and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread_half = (ask - bid) / 2.0
            return (mid + spread_half) if side == "BUY" else (mid - spread_half)
        return ltp

    async def place_order(self, symbol: str, side: str, quantity: int, order_type: str, price: float = 0.0) -> dict[str, Any]:
        fill = self._fill_price(symbol, side)
        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "broker_order_id": f"PAPER_{order_id[:8]}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "average_price": fill,
            "status": "FILLED",
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }
        self._orders.append(order)

        # Track positions
        if side == "BUY":
            self._positions[symbol] = {
                "symbol": symbol, "quantity": quantity,
                "average_price": fill, "pnl": 0.0,
            }
        elif symbol in self._positions:
            pos = self._positions[symbol]
            entry = pos["average_price"]
            pnl = (fill - entry) * quantity
            pos["pnl"] += pnl
            pos["quantity"] -= quantity
            if pos["quantity"] <= 0:
                del self._positions[symbol]

        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        return True  # Paper orders always cancel immediately

    async def get_orders(self) -> list[dict[str, Any]]:
        return list(self._orders)

    async def get_positions(self) -> list[dict[str, Any]]:
        return list(self._positions.values())

    async def get_quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return {sym: self._last_quotes.get(sym, {"ltp": 0.0}) for sym in symbols}

    async def get_profile(self) -> dict[str, Any]:
        return {"user_id": "PAPER_TRADER", "user_name": "Paper Trader", "email": "paper@test.com"}

    async def get_instruments(self, exchange: str = "NFO") -> list[dict[str, Any]]:
        return []
