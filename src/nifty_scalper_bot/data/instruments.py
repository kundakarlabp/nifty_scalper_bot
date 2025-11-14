"""Instrument utilities for resolving trading symbols to instrument tokens."""

# ruff: noqa: I001

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, floor
import os
from typing import Any

from nifty_scalper_bot.infra.metrics import METRICS
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
        self._warned_no_token: set[str] = set()

    def _clear_warning_state(self, symbol: str) -> None:
        """Clear cached warning state for *symbol* if present.

        Args:
            symbol: Trading symbol whose warning cache should be reset.

        Returns:
            None.

        Raises:
            None.
        """

        base = symbol.split(":", 1)[-1].strip().upper()
        if base:
            self._warned_no_token.discard(base)

    def _ingest_instrument_row(self, row: Mapping[str, Any]) -> None:
        """Load a single broker instrument row into resolver caches.

        Args:
            row: Mapping describing broker instrument metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver._ingest_instrument_row",
            extra={
                "event": "instrument_resolver_ingest_enter",
                "keys": sorted(row.keys()),
            },
        )
        try:
            tradingsymbol = str(
                row.get("tradingsymbol") or row.get("symbol") or ""
            ).strip()
            if not tradingsymbol:
                return
            exchange = (row.get("exchange") or "").strip().upper()
            token_value = row.get("instrument_token") or row.get("token")
            if token_value is None:
                return
            try:
                token_int = int(float(token_value))
            except (TypeError, ValueError) as exc:
                log.debug(
                    "instrument_resolver_ingest_token_cast_failed",
                    extra={
                        "event": "instrument_resolver_ingest_token_cast_failed",
                        "tradingsymbol": tradingsymbol,
                        "exchange": exchange,
                        "token_value": token_value,
                    },
                )
                raise BrokerError("Invalid instrument token") from exc
            key = tradingsymbol.upper()
            self._by_symbol.setdefault(key, token_int)
            if exchange:
                self._by_symbol.setdefault(f"{exchange}:{key}", token_int)
            self._symbol_by_token.setdefault(token_int, tradingsymbol)
            if exchange:
                self._exchange_by_token.setdefault(token_int, exchange)
            self._clear_warning_state(key)
            self._clear_warning_state(tradingsymbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver._ingest_instrument_row: %s",
                exc,
                extra={
                    "event": "instrument_resolver_ingest_error",
                    "row_repr": str(dict(row)),
                },
                exc_info=exc,
            )

    def _seed_well_known(self) -> None:
        """Populate resolver caches with baked-in fallbacks."""

        log.debug(
            "Entered InstrumentResolver._seed_well_known",
            extra={"event": "instrument_resolver_seed_enter"},
        )
        for key, value in WELL_KNOWN.items():
            try:
                token_int = int(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - config guard
                log.error(
                    "Failure in InstrumentResolver._seed_well_known: %s",
                    exc,
                    extra={
                        "event": "instrument_resolver_seed_error",
                        "key": key,
                        "value": value,
                    },
                    exc_info=exc,
                )
                continue
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
            self._clear_warning_state(alias_symbol)

        for token, canonical_symbol in CANONICAL_TOKENS.items():
            self._symbol_by_token[token] = canonical_symbol
            self._exchange_by_token.setdefault(token, "NSE")
            self._clear_warning_state(canonical_symbol)

    def upsert(
        self, symbol: str, token: int, *, exchange: str | None = None
    ) -> None:
        """Insert or refresh resolver caches for *symbol* and *token*.

        Args:
            symbol: Trading symbol to associate with the token.
            token: Numeric instrument token supplied by the broker.
            exchange: Optional exchange code (e.g. ``"NFO"``).

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.upsert",
            extra={"event": "instrument_resolver_upsert_enter", "symbol": symbol},
        )
        try:
            normalized = (symbol or "").strip().upper()
            if not normalized:
                log.info(
                    "Condition met: instrument_resolver_upsert_blank",
                    extra={"event": "instrument_resolver_upsert_blank"},
                )
                return
            token_int = int(token)
            base_symbol = normalized.split(":", 1)[-1] or normalized
            self._by_symbol[normalized] = token_int
            self._by_symbol.setdefault(base_symbol, token_int)
            exchange_hint = (exchange or "").strip().upper()
            if exchange_hint:
                self._by_symbol[f"{exchange_hint}:{base_symbol}"] = token_int
                self._exchange_by_token[token_int] = exchange_hint
            self._symbol_by_token[token_int] = base_symbol
            self._clear_warning_state(base_symbol)
            log.info(
                "Condition met: instrument_resolver_upsert",
                extra={
                    "event": "instrument_resolver_upsert",
                    "symbol": base_symbol,
                    "token": token_int,
                    "exchange": exchange_hint or "",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.upsert: %s",
                exc,
                extra={
                    "event": "instrument_resolver_upsert_error",
                    "symbol": symbol,
                    "token": token,
                },
                exc_info=exc,
            )

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
                if not isinstance(row, Mapping):
                    continue
                self._ingest_instrument_row(row)

        self._seed_well_known()

        log.info("InstrumentResolver ready with %d symbols", len(self._by_symbol))

    def warm_from_broker_dump(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Warm caches directly from *rows* produced by broker dumps.

        Args:
            rows: Iterable of broker instrument rows already fetched by caller.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.warm_from_broker_dump",
            extra={"event": "instrument_resolver_warm_dump_enter"},
        )
        try:
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                self._ingest_instrument_row(row)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.warm_from_broker_dump: %s",
                exc,
                extra={"event": "instrument_resolver_warm_dump_error"},
                exc_info=exc,
            )
        self._seed_well_known()

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

    def resolve_symbol_to_token(self, symbol: str) -> int | None:
        """Resolve *symbol* into an instrument token using resolver caches.

        Args:
            symbol: Trading symbol supplied by callers.

        Returns:
            int | None: Instrument token when available, otherwise ``None``.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.resolve_symbol_to_token",
            extra={"event": "instrument_resolver_token_enter", "symbol": symbol},
        )
        try:
            return self.resolve(symbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.resolve_symbol_to_token: %s",
                exc,
                extra={
                    "event": "instrument_resolver_token_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            return None

    def resolve_exchange(self, symbol: str) -> str | None:
        """Return exchange hint for *symbol* when available.

        Args:
            symbol: Trading symbol supplied by callers.

        Returns:
            str | None: Exchange identifier (e.g. ``"NFO"``) when resolvable.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.resolve_exchange",
            extra={"event": "instrument_resolver_exchange_enter", "symbol": symbol},
        )
        try:
            token = self.resolve_symbol_to_token(symbol)
            if token is not None:
                exchange = self._exchange_by_token.get(int(token))
                if exchange:
                    return exchange
            try:
                return self.exchange_for_symbol(symbol)
            except BrokerError:
                return None
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.resolve_exchange: %s",
                exc,
                extra={
                    "event": "instrument_resolver_exchange_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            return None

    def exchange_for(self, symbol: str) -> str | None:
        """Alias providing exchange lookup compatible with legacy callers.

        Args:
            symbol: Trading symbol requiring exchange lookup.

        Returns:
            str | None: Exchange identifier or ``None`` when unavailable.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.exchange_for",
            extra={"event": "instrument_resolver_exchange_for_enter", "symbol": symbol},
        )
        try:
            return self.resolve_exchange(symbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.exchange_for: %s",
                exc,
                extra={
                    "event": "instrument_resolver_exchange_for_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            return None

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
            base_symbol = key.split(":", 1)[-1].strip().upper()
            if base_symbol not in self._warned_no_token:
                log.warning("InstrumentResolver: no token for '%s'", symbol)
                if base_symbol:
                    self._warned_no_token.add(base_symbol)
                    METRICS.record_resolver_miss(symbol=base_symbol)
            else:
                log.debug(
                    "InstrumentResolver: no token for '%s' (suppressed)",
                    symbol,
                )
            return None
        base_symbol = key.split(":", 1)[-1]
        self._clear_warning_state(base_symbol)
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

    def build_quote_keys(self, symbol: str) -> tuple[str, list[str]]:
        """Return canonical quote key candidates for *symbol*.

        Args:
            symbol: Trading symbol provided by caller.

        Returns:
            tuple[str, list[str]]: Normalized symbol and ordered candidate list
            suitable for broker quote APIs.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.build_quote_keys",
            extra={"event": "instrument_resolver_build_keys_enter", "symbol": symbol},
        )
        normalized = (symbol or "").strip()
        if not normalized:
            return "", []
        upper = normalized.upper()
        candidates: list[str] = []
        try:
            exchange_hint = self.resolve_exchange(upper)
            _, tradingsymbol = self._normalize_option_symbol(upper)
        except BrokerError:
            tradingsymbol = upper
            exchange_hint = self.resolve_exchange(upper)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.build_quote_keys: %s",
                exc,
                extra={
                    "event": "instrument_resolver_build_keys_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            tradingsymbol = upper
            exchange_hint = None

        tradingsymbol = tradingsymbol.replace(" ", "")
        if exchange_hint:
            candidates.append(f"{exchange_hint}:{tradingsymbol}")
        candidates.append(tradingsymbol)
        token = self.resolve_symbol_to_token(tradingsymbol)
        if token is not None:
            formatted = self.format_token_as_symbol(token)
            if formatted:
                candidates.append(formatted)
        seen: set[str] = set()
        ordered: list[str] = []
        for key in candidates:
            if key and key not in seen:
                seen.add(key)
                ordered.append(key)
        return tradingsymbol, ordered

    def canonicalize(self, symbol: str) -> tuple[str, str | None, str]:
        """Return canonical symbol, exchange, and segment for *symbol*.

        Args:
            symbol: Trading symbol provided by user input.

        Returns:
            tuple[str, str | None, str]: Canonical symbol, exchange hint, and
            asset segment label.

        Raises:
            None.
        """

        log.debug(
            "Entered InstrumentResolver.canonicalize",
            extra={"event": "instrument_resolver_canonicalize_enter", "symbol": symbol},
        )
        normalized = (symbol or "").strip().upper()
        if not normalized:
            return "", None, "UNKNOWN"
        try:
            exchange, tradingsymbol = self._normalize_option_symbol(normalized)
            exchange = exchange or self.resolve_exchange(tradingsymbol) or "NFO"
            segment = "OPTIONS"
            return tradingsymbol, exchange, segment
        except BrokerError:
            exchange = self.resolve_exchange(normalized)
            segment = "INDEX" if normalized in WELL_KNOWN else "UNKNOWN"
            return normalized, exchange, segment
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in InstrumentResolver.canonicalize: %s",
                exc,
                extra={
                    "event": "instrument_resolver_canonicalize_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            return normalized, None, "UNKNOWN"

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
