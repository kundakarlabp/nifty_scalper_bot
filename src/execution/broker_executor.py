from __future__ import annotations

"""Order executor backed by a :class:`Broker`."""

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.broker.interface import Broker, OrderRequest, OrderType, Side, TimeInForce
from src.utils.env import env_flag


class BrokerOrderExecutor:
    """Lightweight wrapper to submit orders via a broker."""

    def __init__(
        self,
        broker: Broker,
        instrument_id_mapper: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.broker = broker
        self.instrument_id_mapper = instrument_id_mapper

    # ------------------------------------------------------------------
    def place_order(
        self,
        req: OrderRequest | Mapping[str, Any],
        instrument_id_mapper: Optional[Callable[[str], int]] = None,
    ) -> str:
        """Place an order and return the broker order ID."""
        order = req if isinstance(req, OrderRequest) else self._coerce_request(
            req, instrument_id_mapper
        )
        if self._kill_switch_engaged():
            raise RuntimeError("kill switch active")
        return self.broker.place_order(order)

    # ------------------------------------------------------------------
    def buy(self, symbol_or_token: str | int, qty: int, **kwargs: Any) -> str:
        """Convenience helper to submit a buy order."""
        return self._simple_side(Side.BUY, symbol_or_token, qty, **kwargs)

    def sell(self, symbol_or_token: str | int, qty: int, **kwargs: Any) -> str:
        """Convenience helper to submit a sell order."""
        return self._simple_side(Side.SELL, symbol_or_token, qty, **kwargs)

    # ------------------------------------------------------------------
    def _simple_side(
        self, side: Side, symbol_or_token: str | int, qty: int, **kwargs: Any
    ) -> str:
        token = self._resolve_token(symbol_or_token)
        req = OrderRequest(
            instrument_id=token,
            side=side,
            qty=qty,
            order_type=kwargs.get("order_type", OrderType.MARKET),
            price=kwargs.get("price"),
            stop_loss=kwargs.get("stop_loss"),
            target=kwargs.get("target"),
            tif=kwargs.get("tif", TimeInForce.DAY),
            client_order_id=kwargs.get("client_order_id"),
            metadata=kwargs.get("metadata"),
        )
        return self.place_order(req)

    @staticmethod
    def _kill_switch_engaged() -> bool:
        """Return ``True`` when global trading should be halted."""
        if env_flag("KILL_SWITCH", False):
            return True
        path_val = os.environ.get("KILL_SWITCH_FILE")
        if path_val and Path(path_val).exists():
            return True
        return False

    def _resolve_token(self, symbol_or_token: str | int) -> int:
        if isinstance(symbol_or_token, int):
            return symbol_or_token
        mapper = self.instrument_id_mapper
        if mapper is None:
            raise ValueError("instrument_id_mapper not configured")
        token = mapper(symbol_or_token)
        if token is None:
            raise ValueError(f"unknown symbol: {symbol_or_token}")
        return int(token)

    def _coerce_request(
        self,
        payload: Mapping[str, Any],
        mapper_override: Optional[Callable[[str], int]] = None,
    ) -> OrderRequest:
        data = dict(payload)
        token = data.get("instrument_id")
        if token is None:
            symbol = data.get("symbol")
            mapper = mapper_override or self.instrument_id_mapper
            if not (symbol and mapper):
                raise ValueError("instrument_id or symbol required")
            token = mapper(symbol)
        side_val = data.get("side")
        if isinstance(side_val, str):
            side = Side[side_val.upper()]
        elif isinstance(side_val, Side):
            side = side_val
        else:
            raise ValueError("side required")
        order_type = data.get("order_type", OrderType.MARKET)
        if isinstance(order_type, str):
            order_type = OrderType[order_type.upper()]
        tif = data.get("tif", TimeInForce.DAY)
        if isinstance(tif, str):
            tif = TimeInForce[tif.upper()]
        qty_val = data.get("qty")
        if qty_val is None:
            raise ValueError("qty required")
        return OrderRequest(
            instrument_id=int(token),
            side=side,
            qty=int(qty_val),
            order_type=order_type,
            price=data.get("price"),
            stop_loss=data.get("stop_loss"),
            target=data.get("target"),
            tif=tif,
            client_order_id=data.get("client_order_id"),
            metadata=data.get("metadata"),
        )
