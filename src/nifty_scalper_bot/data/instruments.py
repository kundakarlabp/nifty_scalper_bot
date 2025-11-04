"""Instrument utilities for resolving trading symbols to instrument tokens."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Iterable

from nifty_scalper_bot.utils.errors import BrokerError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.options_math import black_scholes_greeks

log = get_logger(__name__)

# Common index tokens (Zerodha). Keep here as fallbacks.
WELL_KNOWN = {
    # NSE Index tokens
    "NIFTY": 256265,  # NIFTY 50 spot
    # Robust aliases people often use
    "NIFTY50": 256265,
    "NIFTY-50": 256265,
    "NSE:NIFTY": 256265,
    "NSE:NIFTY 50": 256265,
    "NIFTY 50": 256265,
    "BANKNIFTY": 260105,  # NIFTY BANK spot
    "NIFTY BANK": 260105,
    "FINNIFTY": 257801,  # NIFTY FIN SERVICE spot
    "MIDCPNIFTY": 288009,  # NIFTY MIDCAP SELECT spot
}

# Canonical trading symbols used when formatting tokens for REST calls.
CANONICAL_TOKENS: dict[int, str] = {
    256265: "NIFTY 50",
    260105: "NIFTY BANK",
    257801: "NIFTY FIN SERVICE",
    288009: "NIFTY MIDCAP SELECT",
}


@dataclass(slots=True)
class Instrument:
    """Lightweight representation of a trading instrument."""

    tradingsymbol: str
    exchange: str | None
    instrument_token: int


class InstrumentResolver:
    """Caches broker instruments and provides reliable symbol resolution."""

    def __init__(self, broker_client: Any) -> None:
        self._broker = broker_client
        self._by_symbol: dict[str, int] = {}
        self._symbol_by_token: dict[int, str] = {}
        self._exchange_by_token: dict[int, str] = {}
        self._option_contracts: dict[str, list[dict[str, Any]]] = {}

    def warm(self) -> None:
        """Load instruments from the broker and seed fallback tokens."""

        items: Iterable[dict[str, Any]] | None = None
        for name in ("list_instruments", "get_instruments", "instruments"):
            fn = getattr(self._broker, name, None)
            if callable(fn):
                try:
                    items = fn()
                except Exception as exc:  # noqa: BLE001
                    log.warning("InstrumentResolver: %s() failed: %s", name, exc)
                break

        # Allow partial environments—don't fail if broker blocks the call.
        if items:
            for row in items:
                ts = str(row.get("tradingsymbol") or row.get("symbol") or "").strip()
                ex = (row.get("exchange") or "").strip().upper()
                tok = row.get("instrument_token") or row.get("token")
                if not ts or tok is None:
                    continue
                key = ts.upper()
                token_int = int(tok)
                self._by_symbol.setdefault(key, token_int)
                if ex:
                    self._by_symbol.setdefault(f"{ex}:{key}", token_int)
                self._symbol_by_token.setdefault(token_int, ts)
                if ex:
                    self._exchange_by_token.setdefault(token_int, ex)

        for key, value in WELL_KNOWN.items():
            token_int = int(value)
            normalized_key = key.upper()
            alias_symbol = normalized_key.split(":", 1)[-1]
            exchange = (
                normalized_key.split(":", 1)[0] if ":" in normalized_key else "NSE"
            )

            self._by_symbol.setdefault(normalized_key, token_int)
            if ":" not in normalized_key:
                self._by_symbol.setdefault(f"{exchange}:{alias_symbol}", token_int)

            self._symbol_by_token.setdefault(token_int, alias_symbol)
            self._exchange_by_token.setdefault(token_int, exchange)

        for token, canonical_symbol in CANONICAL_TOKENS.items():
            self._symbol_by_token[token] = canonical_symbol
            self._exchange_by_token.setdefault(token, "NSE")

        log.info("InstrumentResolver ready with %d symbols", len(self._by_symbol))

    def option_contracts(
        self,
        underlying: str,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return cached option contract metadata for *underlying*.

        The resolver loads option instruments from the broker on demand and
        retains a sanitized cache keyed by normalized underlying (for example,
        ``"NIFTY"``).  When ``force_refresh`` is ``True`` the metadata is
        reloaded from the broker even if a cached copy exists.
        """

        normalized = (underlying or "").strip().upper()
        if not normalized:
            return []
        if not force_refresh:
            cached = self._option_contracts.get(normalized)
            if cached is not None:
                return [dict(entry) for entry in cached]

        contracts = self._load_option_contracts(normalized)
        if contracts:
            self._option_contracts[normalized] = contracts
        elif force_refresh:
            self._option_contracts.pop(normalized, None)

        return [dict(entry) for entry in self._option_contracts.get(normalized, [])]

    def _load_option_contracts(self, underlying: str) -> list[dict[str, Any]]:
        loader: Any | None = None
        loader_name = ""
        for name in (
            "load_instruments",
            "list_instruments",
            "get_instruments",
            "instruments",
        ):
            candidate = getattr(self._broker, name, None)
            if callable(candidate):
                loader = candidate
                loader_name = name
                break
        if loader is None:
            log.warning("InstrumentResolver: no instrument loader for option metadata")
            return []

        try:
            if loader_name == "load_instruments":
                items = loader("NFO")
            else:
                items = loader()
        except TypeError:
            items = loader("NFO")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "InstrumentResolver: %s() failed for option metadata: %s",
                loader_name,
                exc,
            )
            return []

        contracts: list[dict[str, Any]] = []
        for row in items or []:
            if not isinstance(row, dict):
                continue
            tradingsymbol = str(
                row.get("tradingsymbol") or row.get("symbol") or ""
            ).strip()
            if not tradingsymbol:
                continue
            symbol_upper = tradingsymbol.upper()
            name = str(row.get("name") or "").strip().upper()
            if not symbol_upper.startswith(underlying):
                if name != underlying:
                    continue
            option_type = str(
                row.get("instrument_type") or row.get("type") or symbol_upper[-2:]
            ).upper()
            if option_type not in {"CE", "PE"}:
                continue
            raw_token = row.get("instrument_token") or row.get("token")
            token: int | None = None
            if isinstance(raw_token, (int, float)):
                token = int(raw_token)
            elif isinstance(raw_token, str):
                candidate = raw_token.strip()
                if candidate:
                    try:
                        token = int(float(candidate))
                    except (TypeError, ValueError):
                        token = None
            if token is None:
                continue
            expiry = (
                row.get("expiry")
                or row.get("expiry_date")
                or row.get("expiryDate")
                or row.get("expiryDateTime")
            )
            strike_raw = (
                row.get("strike") or row.get("strike_price") or row.get("strikePrice")
            )
            try:
                strike = float(strike_raw) if strike_raw is not None else None
            except (TypeError, ValueError):
                strike = None
            lot_size_raw = row.get("lot_size") or row.get("lotsize")
            try:
                lot_size = int(lot_size_raw) if lot_size_raw is not None else None
            except (TypeError, ValueError):
                lot_size = None
            tick_size_raw = row.get("tick_size") or row.get("ticksize")
            try:
                tick_size = float(tick_size_raw) if tick_size_raw is not None else None
            except (TypeError, ValueError):
                tick_size = None
            contracts.append(
                {
                    "instrument_token": token,
                    "tradingsymbol": tradingsymbol,
                    "option_type": option_type,
                    "expiry": expiry,
                    "strike": strike,
                    "lot_size": lot_size,
                    "tick_size": tick_size,
                    "raw": dict(row),
                }
            )

        return contracts

    @staticmethod
    def _normalize_option_symbol(symbol: str) -> tuple[str | None, str]:
        """Return ``(exchange, tradingsymbol)`` for option contracts."""

        if not symbol:
            raise BrokerError("Symbol required for option resolution")

        raw = symbol.strip().upper()
        if not raw:
            raise BrokerError("Symbol required for option resolution")

        exchange: str | None = None
        tradingsymbol = raw
        if ":" in raw:
            prefix, remainder = raw.split(":", 1)
            exchange = prefix or None
            tradingsymbol = remainder.strip()
        tradingsymbol = tradingsymbol.replace(" ", "")

        if not tradingsymbol:
            raise BrokerError("Tradingsymbol missing for option resolution")

        if tradingsymbol.endswith("FUT"):
            raise BrokerError("Futures disabled for this bot")

        if not tradingsymbol.endswith("CE") and not tradingsymbol.endswith("PE"):
            raise BrokerError("Only NIFTY options (CE/PE) are allowed")

        return exchange, tradingsymbol

    def exchange_for_symbol(self, symbol: str) -> str:
        """Return the required exchange for an option symbol."""

        exchange, _tradingsymbol = self._normalize_option_symbol(symbol)
        if exchange is not None and exchange != "NFO":
            raise BrokerError("Only NFO exchange is supported for NIFTY options")
        return "NFO"

    def tradingsymbol_for_order(self, symbol: str) -> str:
        """Return a sanitized tradingsymbol for order placement."""

        _exchange, tradingsymbol = self._normalize_option_symbol(symbol)
        return tradingsymbol

    def lookup(self, symbol: str) -> dict[str, Any] | None:
        """Return instrument metadata for *symbol* when available.

        Args:
            symbol: Trading symbol or token hint supplied by the caller.

        Returns:
            dict[str, Any] | None: Mapping with token, exchange, and symbol data.

        Raises:
            None.
        """

        normalized = (symbol or "").strip()
        if not normalized:
            return None
        try:
            token = self.resolve(normalized)
            if token is None:
                return None
            token_int = int(token)
            tradingsymbol = self._symbol_by_token.get(token_int)
            if not tradingsymbol:
                tradingsymbol = normalized.split(":", 1)[-1].strip().upper()
            exchange = self._exchange_by_token.get(token_int)
            if not exchange:
                exchange = (
                    normalized.split(":", 1)[0].strip().upper()
                    if ":" in normalized
                    else "NFO"
                )
            return {
                "instrument_token": token_int,
                "tradingsymbol": str(tradingsymbol).upper(),
                "exchange": str(exchange).upper(),
            }
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.lookup: %s",
                exc,
                extra={
                    "event": "instrument_lookup_failed",
                    "symbol": normalized,
                    "error": str(exc),
                },
            )
            return None

    def resolve(self, symbol: str) -> int | None:
        """Resolve *symbol* into an instrument token if available."""

        if not symbol:
            return None
        key = symbol.strip().upper()
        token = self._by_symbol.get(key)
        if token is None:
            token = self._by_symbol.get(f"NSE:{key}")
        if token is None:
            token = self._by_symbol.get(f"BSE:{key}")
        if token is None:
            log.warning("InstrumentResolver: no token for '%s'", symbol)
            return None
        return int(token)

    def resolve_many(self, symbols: list[str]) -> list[int]:
        """Resolve multiple *symbols* into instrument tokens."""

        tokens: list[int] = []
        for symbol in symbols:
            token = self.resolve(symbol)
            if token is not None:
                tokens.append(int(token))
        return tokens

    def resolve_token_to_symbol(self, token: int) -> str | None:
        """Return the trading symbol for *token* if known."""

        return self._symbol_by_token.get(int(token))

    def format_token_as_symbol(self, token: int) -> str | None:
        """Return ``EXCHANGE:SYMBOL`` formatted for Zerodha REST calls."""

        token_int = int(token)
        symbol = self._symbol_by_token.get(token_int)
        if not symbol:
            return None
        exchange = self._exchange_by_token.get(token_int) or "NSE"
        return f"{exchange}:{symbol.upper()}"

    def lot_size_for_symbol(self, symbol: str) -> int:
        """Return configured lot size for *symbol* with fallbacks."""

        sym = (symbol or "").split(":")[-1].strip().upper()
        cache = getattr(self, "_instruments_cache", None)
        if isinstance(cache, dict):
            meta = cache.get(sym) or cache.get(sym.replace("NFO:", ""))
            if isinstance(meta, dict):
                lot_size = meta.get("lot_size")
                if isinstance(lot_size, int) and lot_size > 0:
                    return lot_size
        try:
            return max(1, int(os.getenv("INSTRUMENTS__NIFTY_LOT_SIZE", "75")))
        except Exception:  # pragma: no cover - defensive
            return 75


def atm_strike_for_spot(spot: float, step: int = 50) -> int:
    """Return the ATM strike rounded to the nearest *step*."""

    if step <= 0:
        raise ValueError("step must be positive")
    strike = round(float(spot) / step) * step
    return int(strike)


def strike_for_delta(
    spot: float,
    iv: float,
    ttm: float,
    target_delta: float,
    *,
    call: bool,
    step: int = 50,
    rate: float = 0.06,
) -> int:
    """Return strike approximating *target_delta* using binary search."""

    if step <= 0:
        raise ValueError("step must be positive")
    low = max(step, floor(spot / step - 10) * step)
    high = max(step, ceil(spot / step + 10) * step)
    best_strike = atm_strike_for_spot(spot, step)
    best_diff = float("inf")
    for _ in range(32):
        mid = (low + high) // 2
        greeks = black_scholes_greeks(spot, float(mid), ttm, rate, iv, is_call=call)
        delta = greeks.get("delta", 0.0)
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = mid
        if call:
            if delta > target_delta:
                high = mid - step
            else:
                low = mid + step
        else:
            if delta < -target_delta:
                high = mid - step
            else:
                low = mid + step
        if high < low:
            break
    return int(best_strike)


__all__ = [
    "Instrument",
    "InstrumentResolver",
    "WELL_KNOWN",
    "atm_strike_for_spot",
    "strike_for_delta",
]
