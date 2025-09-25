"""Simplified scalper orchestration with confirmation and risk guards."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

from src.data.market_data import get_best_ask
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.utils.helpers import get_next_thursday

log = logging.getLogger(__name__)


PriceFetcher = Callable[[str], float]
ExpiryResolver = Callable[[], str]


def _round_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    factor = round(1.0 / tick_size)
    return round(price * factor) / factor


@dataclass(slots=True)
class ScalperStrategy:
    """Coordinate a CE/PE entry with confirmation and risk protections."""

    order_manager: OrderManager
    risk_manager: RiskManager
    underlying: str = "NIFTY"
    tick_size: float = 0.05
    price_fetcher: PriceFetcher = get_best_ask
    expiry_resolver: ExpiryResolver = get_next_thursday
    _side_map: Mapping[str, str] = field(
        default_factory=lambda: {"BUY": "SELL", "SELL": "BUY"}, init=False
    )

    def build_option_symbols(self, strike: int | str) -> Dict[str, str]:
        """Return CE/PE trading symbols for the upcoming weekly expiry."""

        expiry = self.expiry_resolver()
        try:
            strike_val = int(strike)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            raise ValueError("strike must be numeric")
        strike_fmt = f"{strike_val:05d}" if strike_val < 100000 else str(strike_val)
        base = self.underlying.upper()
        return {
            "expiry": expiry,
            "ce": f"{base}{expiry}{strike_fmt}CE",
            "pe": f"{base}{expiry}{strike_fmt}PE",
        }

    def _fetch_price(self, symbol: str) -> float:
        quote_symbol = symbol if symbol.startswith("NFO:") else f"NFO:{symbol}"
        raw = float(self.price_fetcher(quote_symbol))
        price = _round_to_tick(raw, self.tick_size)
        if price <= 0:
            raise ValueError(f"invalid price for {symbol}: {price}")
        return price

    def trade_straddle(
        self,
        strike: int | str,
        *,
        quantity: int,
        atr: float,
        side: str = "BUY",
    ) -> Dict[str, Any]:
        """Place CE/PE legs and enforce confirmation safeguards."""

        if quantity <= 0:
            raise ValueError("quantity must be positive")
        meta = self.build_option_symbols(strike)
        ce_symbol = meta["ce"]
        pe_symbol = meta["pe"]
        expiry = meta["expiry"]

        ce_price = self._fetch_price(ce_symbol)
        pe_price = self._fetch_price(pe_symbol)
        risk_side = "LONG" if side.upper() == "BUY" else "SHORT"
        ce_sl = self.risk_manager.calculate_stop_loss(ce_price, atr, side=risk_side)
        pe_sl = self.risk_manager.calculate_stop_loss(pe_price, atr, side=risk_side)

        order_side = side.upper()
        ce_params = {
            "symbol": ce_symbol,
            "transaction_type": order_side,
            "order_type": "LIMIT",
            "quantity": int(quantity),
            "price": ce_price,
        }
        pe_params = {
            "symbol": pe_symbol,
            "transaction_type": order_side,
            "order_type": "LIMIT",
            "quantity": int(quantity),
            "price": pe_price,
        }

        ce_order_id = self.order_manager.place_order_with_confirmation(ce_params)
        pe_order_id = self.order_manager.place_order_with_confirmation(pe_params)

        result = {
            "expiry": expiry,
            "ce_symbol": ce_symbol,
            "pe_symbol": pe_symbol,
            "ce_price": ce_price,
            "pe_price": pe_price,
            "ce_stop_loss": ce_sl,
            "pe_stop_loss": pe_sl,
            "ce_order_id": ce_order_id,
            "pe_order_id": pe_order_id,
        }

        ce_filled = ce_order_id is not None
        pe_filled = pe_order_id is not None
        close_side = self._side_map.get(order_side, "SELL")

        if ce_filled and pe_filled:
            result["status"] = "complete"
            log.info(
                "Placed straddle at strike %s (CE=%s, PE=%s)",
                strike,
                ce_price,
                pe_price,
            )
            return result

        if ce_filled and not pe_filled:
            self.order_manager.square_off_position(
                ce_symbol, side=close_side, quantity=quantity
            )
            log.critical("Partial fill detected; squared off CE leg for %s", ce_symbol)
            result["status"] = "partial_ce"
            return result

        if pe_filled and not ce_filled:
            self.order_manager.square_off_position(
                pe_symbol, side=close_side, quantity=quantity
            )
            log.critical("Partial fill detected; squared off PE leg for %s", pe_symbol)
            result["status"] = "partial_pe"
            return result

        log.error("Both legs failed for strike %s; aborting straddle", strike)
        result["status"] = "failed"
        return result


__all__ = ["ScalperStrategy"]

