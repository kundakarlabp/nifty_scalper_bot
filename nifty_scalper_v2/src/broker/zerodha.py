"""Zerodha Kite Connect broker implementation."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .base import BrokerGateway

if TYPE_CHECKING:
    from ..config.settings import BrokerSettings

log = logging.getLogger(__name__)


class ZerodhaBroker(BrokerGateway):
    """Wraps kiteconnect.KiteConnect for async use via httpx."""

    def __init__(self, settings: "BrokerSettings") -> None:
        self._settings = settings
        self._kite = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize Kite Connect and verify credentials."""
        try:
            from kiteconnect import KiteConnect  # type: ignore[import]
            self._kite = KiteConnect(
                api_key=self._settings.api_key.get_secret_value()
            )
            self._kite.set_access_token(
                self._settings.access_token.get_secret_value()
            )
            # Verify by fetching profile
            await self.get_profile()
            self._connected = True
            log.info("Zerodha Kite connected successfully")
        except Exception as e:
            log.error("Kite connection failed: %s", e)
            raise

    async def place_order(self, symbol: str, side: str, quantity: int, order_type: str, price: float = 0.0) -> dict[str, Any]:
        if not self._kite:
            raise RuntimeError("Broker not connected")
        import asyncio
        loop = asyncio.get_event_loop()
        kite_order_type = "MARKET" if order_type == "MARKET" else "LIMIT"
        result = await loop.run_in_executor(None, lambda: self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange=self._kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=self._kite.TRANSACTION_TYPE_BUY if side == "BUY" else self._kite.TRANSACTION_TYPE_SELL,
            quantity=quantity,
            product=self._kite.PRODUCT_MIS,
            order_type=kite_order_type,
            price=price if kite_order_type == "LIMIT" else None,
        ))
        return {"order_id": str(result), "broker_order_id": str(result), "average_price": price, "status": "SUBMITTED"}

    async def cancel_order(self, broker_order_id: str) -> bool:
        if not self._kite:
            return False
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: self._kite.cancel_order(
                variety=self._kite.VARIETY_REGULAR,
                order_id=broker_order_id,
            ))
            return True
        except Exception as e:
            log.error("Cancel failed %s: %s", broker_order_id, e)
            return False

    async def get_orders(self) -> list[dict[str, Any]]:
        if not self._kite:
            return []
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._kite.orders)

    async def get_positions(self) -> list[dict[str, Any]]:
        if not self._kite:
            return []
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._kite.positions)
        return result.get("day", [])

    async def get_quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        if not self._kite:
            return {}
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._kite.quote(symbols))

    async def get_profile(self) -> dict[str, Any]:
        if not self._kite:
            raise RuntimeError("Broker not connected")
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._kite.profile)

    async def get_instruments(self, exchange: str = "NFO") -> list[dict[str, Any]]:
        if not self._kite:
            return []
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._kite.instruments(exchange))
