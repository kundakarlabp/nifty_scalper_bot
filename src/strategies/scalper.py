"""Simplified scalper orchestration with confirmation and risk guards."""
from __future__ import annotations

from datetime import datetime, time, timezone
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping
from zoneinfo import ZoneInfo

from src.config import settings
from src.data.market_data import get_best_ask
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.utils.helpers import get_weekly_expiry

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_STALENESS_MS = 30_000
MAX_DATA_STALENESS_MS = int(
    getattr(
        settings,
        "MAX_DATA_STALENESS_MS",
        getattr(settings, "max_data_staleness_ms", _DEFAULT_STALENESS_MS),
    )
)


PriceFetcher = Callable[[str], float]
ExpiryResolver = Callable[[], str]


def _is_market_hours(now: datetime | None = None) -> bool:
    """Return ``True`` when trading is allowed for the NSE cash window."""

    if getattr(settings, "allow_offhours_testing", False):
        return True

    current = (now or datetime.now(IST)).astimezone(IST)
    if current.weekday() >= 5:
        return False

    return time(9, 15) <= current.time() <= time(15, 25)


class StaleMarketDataError(RuntimeError):
    """Raised when a trade is attempted with stale market data."""


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
    market_data: Any | None = None
    tick_size: float = 0.05
    price_fetcher: PriceFetcher = get_best_ask
    expiry_resolver: ExpiryResolver = get_weekly_expiry
    max_quote_age_seconds: float = 30.0
    _side_map: Mapping[str, str] = field(
        default_factory=lambda: {"BUY": "SELL", "SELL": "BUY"}, init=False
    )

    def _extract_market_timestamp(self) -> datetime | None:
        """Return the freshest known market-data timestamp if available."""

        if self.market_data is None:
            return None

        candidates = (
            "last_tick_dt",
            "last_tick_ts",
            "last_trade_time",
            "last_trade_at",
            "last_quote_time",
            "last_quote_at",
        )

        for attr in candidates:
            value = getattr(self.market_data, attr, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            ts = self._coerce_datetime(value)
            if ts is not None:
                return ts
        return None

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        """Attempt to coerce ``value`` to an aware UTC ``datetime``."""

        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            seconds = float(value)
            if seconds > 1e12:
                seconds /= 1000.0
            if seconds > 1e10:
                return None
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        if isinstance(value, str):
            try:
                normalized = value.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    def _ensure_market_data_fresh(self) -> None:
        """Raise :class:`StaleMarketDataError` if market data is stale."""

        threshold = float(self.max_quote_age_seconds)
        if threshold <= 0:
            return
        last_ts = self._extract_market_timestamp()
        if last_ts is None:
            return
        now = datetime.now(timezone.utc)
        age = (now - last_ts).total_seconds()
        if age < 0:
            return
        if age > threshold:
            logger.warning(
                "Rejecting straddle trade: market data stale (age=%.1fs, limit=%.1fs)",
                age,
                threshold,
            )
            raise StaleMarketDataError(
                f"market data stale (age={age:.1f}s > {threshold:.1f}s)"
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

    def execute_trade(
        self,
        strike: int | str,
        *,
        quantity: int,
        atr: float,
        side: str = "BUY",
    ) -> Dict[str, Any]:
        """Execute a straddle trade after verifying market data freshness."""

        if not _is_market_hours():
            logger.info("Outside market hours. Skipping trade.")
            return {
                "status": "skipped",
                "reason": "off_hours",
            }

        max_age_ms = MAX_DATA_STALENESS_MS
        age_ms = None
        if self.market_data is not None:
            age_ms = getattr(self.market_data, "last_tick_age_ms", None)

        if age_ms is None or age_ms > max_age_ms:
            logger.warning(
                "Data stale (%s ms). Skipping trade.",
                age_ms,
            )
            return {
                "status": "skipped",
                "reason": "data_stale",
                "last_tick_age_ms": age_ms,
            }

        return self.trade_straddle(
            strike,
            quantity=quantity,
            atr=atr,
            side=side,
        )

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
        self._ensure_market_data_fresh()
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
            logger.info(
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
            logger.critical("Partial fill detected; squared off CE leg for %s", ce_symbol)
            result["status"] = "partial_ce"
            return result

        if pe_filled and not ce_filled:
            self.order_manager.square_off_position(
                pe_symbol, side=close_side, quantity=quantity
            )
            logger.critical("Partial fill detected; squared off PE leg for %s", pe_symbol)
            result["status"] = "partial_pe"
            return result

        logger.error("Both legs failed for strike %s; aborting straddle", strike)
        result["status"] = "failed"
        return result


__all__ = ["ScalperStrategy", "StaleMarketDataError"]

