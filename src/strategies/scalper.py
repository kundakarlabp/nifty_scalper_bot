"""Simplified scalper orchestration with confirmation and risk guards."""
from __future__ import annotations

from datetime import datetime, time, timezone
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from zoneinfo import ZoneInfo

from config import settings
from src.data.market_data import get_best_ask
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.utils.helpers import get_weekly_expiry

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_STALENESS_MS = getattr(settings, "max_data_staleness_ms", 30_000)
MAX_DATA_STALENESS_MS = int(
    getattr(settings, "MAX_DATA_STALENESS_MS", _DEFAULT_STALENESS_MS)
)


PriceFetcher = Callable[[str], float]
ExpiryResolver = Callable[[], str]


def _available_cash_equity(kite: Any) -> float:
    """Return a conservative estimate of available cash for NFO trades."""

    try:
        margins = kite.margins("equity")
        available = margins.get("available", {}) or {}
        cash = float(available.get("cash") or 0.0)
        adhoc = float(available.get("adhoc_margin") or 0.0)
        live_balance = float(available.get("live_balance") or 0.0)
        total = cash + adhoc
        return total if total > 0 else live_balance
    except Exception as exc:  # pragma: no cover - network defensive guard
        logger.warning("margin_fetch_failed: %r", exc)
        return 0.0


def _straddle_required_margin_per_lot(
    kite: Any, ce_symbol: str, pe_symbol: str, lot_size: int
) -> float:
    """Ask the broker for the margin impact of a single short straddle."""

    legs = [
        {
            "exchange": "NFO",
            "tradingsymbol": ce_symbol,
            "transaction_type": "SELL",
            "variety": "regular",
            "product": "NRML",
            "order_type": "MARKET",
            "quantity": lot_size,
        },
        {
            "exchange": "NFO",
            "tradingsymbol": pe_symbol,
            "transaction_type": "SELL",
            "variety": "regular",
            "product": "NRML",
            "order_type": "MARKET",
            "quantity": lot_size,
        },
    ]

    try:  # pragma: no cover - network defensive guard
        if hasattr(kite, "order_margins"):
            response = kite.order_margins(legs)
        else:
            response = kite.basket_order_margins(legs, mode="compact")

        if isinstance(response, dict):
            aggregate = (
                response.get("total")
                or response.get("initial_margin")
                or response.get("final")
                or response.get("net")
            )
            if aggregate:
                return float(aggregate)
            orders = response.get("orders") or []
        else:
            orders = response or []

        required = 0.0
        for order in orders:
            required += float(
                order.get("total")
                or order.get("required_margin")
                or order.get("initial_margin")
                or order.get("final")
                or order.get("net")
                or 0.0
            )
        return required
    except Exception as exc:  # pragma: no cover - network defensive guard
        logger.warning("basket_margin_failed: %r", exc)
        return 120_000.0


def _is_market_hours(now: datetime | None = None) -> bool:
    """Check if within Nifty market hours (9:15 AM - 3:25 PM IST)."""

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
    max_quote_age_seconds: float = field(
        default_factory=lambda: MAX_DATA_STALENESS_MS / 1000.0
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

    def _fetch_price(self, symbol: str, *, order_side: str | None = None) -> float:
        """Return the desired limit price for ``symbol``.

        When ``order_side`` is ``"BUY"`` and :attr:`market_data` exposes a
        ``get_marketable_ask`` helper, the method prefers that quote to cross
        the spread.  For other cases the configured :attr:`price_fetcher` is
        used directly to preserve the historical behaviour relied on in tests.
        """

        quote_symbol = symbol if symbol.startswith("NFO:") else f"NFO:{symbol}"
        raw_price: float | None = None

        if (
            order_side is not None
            and order_side.upper() == "BUY"
            and self.market_data is not None
        ):
            getter = getattr(self.market_data, "get_marketable_ask", None)
            if callable(getter):
                try:
                    raw_price = float(getter(symbol))
                except Exception:  # pragma: no cover - defensive guard
                    logger.warning(
                        "marketable ask unavailable for %s; falling back to price fetcher",
                        symbol,
                    )

        if raw_price is None:
            raw_price = float(self.price_fetcher(quote_symbol))

        price = _round_to_tick(raw_price, self.tick_size)
        if price <= 0:
            raise ValueError(f"invalid price for {symbol}: {price}")
        return price

    def _market_data_age_ms(self) -> float | None:
        """Return the reported ``last_tick_age_ms`` if available."""

        if self.market_data is None:
            return None

        age = getattr(self.market_data, "last_tick_age_ms", None)
        if callable(age):
            try:
                age = age()
            except TypeError:  # pragma: no cover - defensive guard
                return None

        if age is None:
            return None

        try:
            return float(age)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None

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
            logger.info("Outside market hours. Skipping.")
            return {
                "status": "skipped",
                "reason": "off_hours",
            }

        age_ms = self._market_data_age_ms()
        if age_ms is None:
            age_ms = getattr(self.market_data, "last_tick_age_ms", None)

        if age_ms is None:
            age_ms = float(MAX_DATA_STALENESS_MS + 1)

        if age_ms > MAX_DATA_STALENESS_MS:
            logger.warning(
                "Data stale (%d ms). Skipping trade.",
                int(age_ms),
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
        qty = int(quantity)
        self._ensure_market_data_fresh()
        meta = self.build_option_symbols(strike)
        ce_symbol = meta["ce"]
        pe_symbol = meta["pe"]
        expiry = meta["expiry"]

        kite_client = getattr(self.order_manager, "kite", None)
        if kite_client is not None:
            lot_size_default = int(getattr(settings, "LOT_SIZE_DEFAULT", 75) or 75)
            lot_size = max(lot_size_default, 1)
            per_lot_margin = _straddle_required_margin_per_lot(
                kite_client, ce_symbol, pe_symbol, lot_size
            )
            available_margin = _available_cash_equity(kite_client)
            lots_required = max(1, math.ceil(qty / lot_size))
            required_margin = per_lot_margin * lots_required
            if available_margin < required_margin:
                logger.warning(
                    (
                        "Insufficient margin for straddle: need ₹%.0f, have ₹%.0f "
                        "(per_lot=₹%.0f, lots=%d)."
                    ),
                    required_margin,
                    available_margin,
                    per_lot_margin,
                    lots_required,
                )
                return {
                    "status": "skipped",
                    "reason": "insufficient_margin",
                    "required_margin": required_margin,
                    "available_margin": available_margin,
                    "per_lot_margin": per_lot_margin,
                    "lots": lots_required,
                    "ce_symbol": ce_symbol,
                    "pe_symbol": pe_symbol,
                }

        ce_price = self._fetch_price(ce_symbol, order_side=side)
        pe_price = self._fetch_price(pe_symbol, order_side=side)
        risk_side = "LONG" if side.upper() == "BUY" else "SHORT"
        ce_sl = self.risk_manager.calculate_stop_loss(ce_price, atr, side=risk_side)
        pe_sl = self.risk_manager.calculate_stop_loss(pe_price, atr, side=risk_side)

        order_side = side.upper()
        ce_params = {
            "exchange": "NFO",
            "symbol": ce_symbol,
            "tradingsymbol": ce_symbol,
            "transaction_type": order_side,
            "order_type": "LIMIT",
            "quantity": qty,
            "price": ce_price,
            "product": "MIS",
        }
        pe_params = {
            "exchange": "NFO",
            "symbol": pe_symbol,
            "tradingsymbol": pe_symbol,
            "transaction_type": order_side,
            "order_type": "LIMIT",
            "quantity": qty,
            "price": pe_price,
            "product": "MIS",
        }

        success = self.order_manager.place_straddle_orders(ce_params, pe_params)
        if success:
            logger.info("✅ Straddle orders placed successfully")
        else:
            logger.warning("❌ Straddle placement failed")

        result = {
            "expiry": expiry,
            "ce_symbol": ce_symbol,
            "pe_symbol": pe_symbol,
            "ce_price": ce_price,
            "pe_price": pe_price,
            "ce_stop_loss": ce_sl,
            "pe_stop_loss": pe_sl,
        }

        if success:
            result["status"] = "complete"
            logger.info(
                "Placed straddle at strike %s (CE=%s, PE=%s)",
                strike,
                ce_price,
                pe_price,
            )
        else:
            result["status"] = "failed"
            logger.error("Straddle placement failed for strike %s", strike)

        return result


__all__ = ["ScalperStrategy", "StaleMarketDataError"]

