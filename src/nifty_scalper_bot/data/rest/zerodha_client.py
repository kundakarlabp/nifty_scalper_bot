"""Production ready Zerodha Kite REST and websocket clients.

Runtime role:
- Low-level Zerodha REST adapter for instruments, quotes, orders, and history.
- Provides broker data to owner modules.
- Must not own live contract selection."""

from __future__ import annotations

from contextlib import suppress
import csv
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    TypeVar,
    cast,
)
import uuid

import httpx

try:
    from kiteconnect import KiteTicker
except ImportError:  # pragma: no cover - compatibility for stubs/older versions
    KiteTicker = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kiteconnect import KiteTicker as KiteTickerType
else:  # pragma: no cover - fallback type when kiteconnect missing
    KiteTickerType = Any

# InstrumentManager is used as the resolver at runtime (typed as Any).
InstrumentResolver = Any  # type: ignore[misc]

from nifty_scalper_bot.data.rest.client import BaseBrokerClient
from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy
from nifty_scalper_bot.utils.env import get_float, get_int, get_str
from nifty_scalper_bot.utils.errors import (
    BrokerAuthenticationError,
    BrokerBalanceUnavailableError,
    BrokerError,
    ConfigurationError,
    OrderPlacementError,
    WebSocketError,
)
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.rate_limiter import RateLimiter, RateLimitError
from nifty_scalper_bot.utils.retry import (
    RetryableError,
    RetryErrorContext,
    retry_with_backoff,
)

T = TypeVar("T")

LOGGER = get_logger(__name__)
_BROKER_SYNC_LOCK = threading.Lock()
KITE_EXCHANGE_TIMEZONE = ZoneInfo("Asia/Kolkata")
KITE_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_kite_datetime(value: datetime | str) -> str:
    """Format Kite historical request datetimes in exchange-local wall time.

    Kite expects timezone-naive date strings for the Indian exchange session.
    Aware datetimes are converted to Asia/Kolkata before formatting. Existing
    naive datetimes are interpreted as already exchange-local to preserve legacy
    callers that intentionally pass local wall-clock values. Non-empty strings
    are preserved because KiteConnect accepts the same serialized form.
    """
    if isinstance(value, str):
        formatted = value.strip()
        if not formatted:
            raise BrokerError("Historical datetime string must be non-empty")
        return formatted
    if not isinstance(value, datetime):
        raise BrokerError("Historical datetime must be a datetime or non-empty string")
    local_value = value
    if value.tzinfo is not None and value.utcoffset() is not None:
        local_value = value.astimezone(KITE_EXCHANGE_TIMEZONE)
    return local_value.strftime(KITE_DATETIME_FORMAT)


def _normalize_historical_token(instrument_token: object) -> int:
    """Return a positive integer Kite historical token or raise BrokerError."""
    try:
        token = int(instrument_token)
    except (TypeError, ValueError) as exc:
        raise BrokerError("Historical instrument token must be a positive integer") from exc
    if token <= 0:
        raise BrokerError("Historical instrument token must be a positive integer")
    return token


@dataclass(frozen=True)
class _RestCacheEntry:
    payload: Any
    updated_at: float

    def is_fresh(self, ttl_sec: float, now: float) -> bool:
        """Check cache freshness. Args: ttl_sec, now. Returns: bool. Raises: None."""

        return now - self.updated_at <= ttl_sec


def _copy_cache_payload(payload: Any) -> Any:
    """Copy cache payload safely. Args: payload. Returns: Any. Raises: None."""

    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        return dict(payload)
    return payload


def _sanitize_access_token(token: str) -> str:
    """Return the plain access token without embedded API key segments."""

    cleaned = (token or "").strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[-1].strip()
    return cleaned


class ZerodhaKiteClient(BaseBrokerClient):
    """Zerodha Kite API client implementing BaseBrokerClient protocol."""

    _DEFAULT_BASE_URL = "https://api.kite.trade"
    _DEFAULT_EXCHANGE = "NSE"
    _QUOTE_BUCKET = "zerodha.quotes"
    _ORDER_BUCKET = "zerodha.orders"
    _HISTORICAL_BUCKET = "zerodha.historical"
    _GENERAL_BUCKET = "zerodha.general"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        access_token: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        limiter: RateLimiter | None = None,
        default_exchange: str = _DEFAULT_EXCHANGE,
        timeout: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Kite client with credentials."""

        # Support both ZERODHA_* and KITE_* environment variable names for backward
        # compatibility.
        self._api_key = (
            api_key
            or os.getenv("ZERODHA_API_KEY")
            or os.getenv("BROKER_API_KEY")
            or os.getenv("KITE_API_KEY")
        )
        self._api_secret = (
            api_secret
            or os.getenv("ZERODHA_API_SECRET")
            or os.getenv("BROKER_API_SECRET")
            or os.getenv("KITE_API_SECRET")
        )
        raw_access_token = (
            access_token
            or os.getenv("ZERODHA_ACCESS_TOKEN")
            or os.getenv("BROKER_ACCESS_TOKEN")
            or os.getenv("KITE_ACCESS_TOKEN")
            or ""
        )
        self._access_token = _sanitize_access_token(raw_access_token)

        if not self._api_key or not self._access_token:
            raise ConfigurationError(
                "Zerodha credentials are not configured. "
                "Set ZERODHA_API_KEY (or KITE_API_KEY) and "
                "ZERODHA_ACCESS_TOKEN (or KITE_ACCESS_TOKEN) environment variables."
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout, read=timeout, connect=2.0)
        configured_retries = get_int("BROKER_RETRIES", default=max_retries)
        self._max_retries = max(1, configured_retries)
        self._default_exchange = default_exchange
        alt_hosts_raw = os.getenv("ZERODHA_ALT_BASE_URLS", "")
        alt_hosts: list[str] = []
        for token in alt_hosts_raw.split(","):
            candidate = token.strip().rstrip("/")
            if candidate:
                alt_hosts.append(candidate)
        host_cycle: list[str] = []
        for candidate in [self._base_url, *alt_hosts]:
            if candidate and candidate not in host_cycle:
                host_cycle.append(candidate)
        self._base_urls: tuple[str, ...] = (
            tuple(host_cycle) if host_cycle else (self._base_url,)
        )
        self._base_index = 0
        self._client = self._create_http_client(self._base_urls[self._base_index])
        self._transient_retry_bonus = 2 if len(self._base_urls) > 1 else 0

        self._limiter = limiter or RateLimiter()
        if limiter is None:
            self._configure_rate_limits()

        # instrument cache: exchange -> mapping of many normalized keys -> row
        self._instrument_cache: dict[str, dict[str, dict[str, Any]]] = {}
        # token_map: instrument_token (int) -> instrument row dict, populated from
        # NFO (and NSE) on each load_instruments("NFO") / load_instruments("NSE") call.
        # Provides O(1) reverse lookup: token -> full instrument metadata.
        self.token_map: dict[int, dict[str, Any]] = {}
        self._resolver: InstrumentResolver | None = None
        self._log_time_fn: Callable[[], float] = time.time
        self._resilience_lock = threading.RLock()
        self._transient_error_streak = 0
        self._breaker_open_until = 0.0
        self._backoff_base = max(
            0.05, float(os.getenv("TICK_BACKOFF_BASE_SECONDS", "0.5"))
        )
        self._backoff_cap = max(
            self._backoff_base, float(os.getenv("TICK_BACKOFF_CAP_SECONDS", "10"))
        )
        self._breaker_threshold = max(
            1, int(os.getenv("TICK_MAX_CONSECUTIVE_ERRORS", "8"))
        )
        self._breaker_cooldown_sec = min(self._backoff_cap * 3.0, 30.0)
        self._log_cooldown_sec = max(
            0.0, float(os.getenv("TICK_ERROR_LOG_COOLDOWN_SEC", "10"))
        )
        self._last_transient_log = 0.0
        self._retry_base_delay = max(
            0.05, get_float("BROKER_BASE_DELAY_SEC", default=self._backoff_base)
        )
        self._retry_max_delay = max(
            self._retry_base_delay,
            get_float("BROKER_MAX_DELAY_SEC", default=self._backoff_cap),
        )
        jitter_value = get_float("BROKER_JITTER", default=0.25)
        self._retry_jitter = max(0.0, min(jitter_value, 1.0))
        margin_segment = (
            get_str("ZERODHA_MARGIN_SEGMENT", "BROKER_MARGIN_SEGMENT", default="equity")
            or "equity"
        )
        normalized_segment = margin_segment.strip().lower() or "equity"
        if normalized_segment not in {"equity", "commodity"}:
            normalized_segment = "equity"
        self._default_margin_segment = normalized_segment
        self._log_throttle_interval = 60.0
        self._last_log_ltp_bulk = 0.0
        self._last_log_margins = 0.0
        self._last_log_balance = 0.0
        self._last_balance_snapshot: dict[str, float] | None = None
        self._last_balance_snapshot_at: float = 0.0
        self._last_balance_success_log_ts = 0.0
        self._last_balance_success_snapshot = None
        self._last_log_instrument_load = 0.0
        self._rest_cache_ttl = max(
            1.0, get_float("BROKER_REST_CACHE_TTL_SEC", default=15.0)
        )
        self._positions_cache: _RestCacheEntry | None = None
        self._orders_cache: _RestCacheEntry | None = None
        self._margins_cache: dict[str, _RestCacheEntry] = {}
        self._auth_invalid = False
        self._auth_invalid_reason: str | None = None
        self._auth_invalid_at: float | None = None
        # Auth latch self-heal: while latched, one request per interval is
        # allowed through as a re-probe so recovery is automatic once the
        # operator fixes the Kite console allowlist/token (no restart needed).
        self._auth_reprobe_interval: float = 60.0
        self._auth_reprobe_next: float = 0.0
        self._auth_failure_generation = 0
        self._auth_failure_alerted_generation = -1
        self._auth_failure_callback: Callable[[dict[str, Any]], None] | None = None
        self._quote_api_available = True
        self._quote_api_error: str | None = None
        self._quote_api_last_checked_at: float | None = None
        self._md_policy = MarketDataPolicy.from_env()
        self._last_quote_error_log_at: dict[str, float] = {}

    @property
    def auth_invalid(self) -> bool:
        """Return whether terminal broker authentication failure is latched."""
        return bool(self._auth_invalid)

    def authentication_status_snapshot(self) -> dict[str, Any]:
        """Return sanitized broker authentication latch state."""
        return {
            "valid": not self._auth_invalid,
            "reason": self._auth_invalid_reason,
            "invalid_at": self._auth_invalid_at,
            "generation": self._auth_failure_generation,
        }

    def set_auth_failure_callback(
        self, callback: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Register callback invoked when terminal authentication failure latches."""
        self._auth_failure_callback = callback

    @staticmethod
    def _is_authentication_failure(
        *,
        status_code: int | None,
        payload: Mapping[str, Any] | None,
        error_text: str,
    ) -> bool:
        """Classify terminal Zerodha authentication/session failures."""
        if status_code in {401, 403}:
            return True
        fragments: list[str] = [error_text or ""]
        if isinstance(payload, Mapping):
            for key in ("message", "error_type", "status", "error"):
                value = payload.get(key)
                if value is not None:
                    fragments.append(str(value))
        text = " ".join(fragments).lower()
        if isinstance(payload, Mapping):
            error_type = str(payload.get("error_type") or "").strip().lower()
            if error_type in {
                "tokenexception",
                "sessionexception",
                "authenticationexception",
            }:
                return True
            if error_type == "inputexception":
                return False
        auth_tokens = (
            "incorrect api_key",
            "incorrect access_token",
            "invalid access_token",
            "invalid session",
            "session expired",
            "token expired",
            "authentication failed",
            "unauthorized",
        )
        return any(token in text for token in auth_tokens)

    def _clear_rest_caches(self) -> None:
        self._positions_cache = None
        self._orders_cache = None
        self._margins_cache.clear()

    def _mark_authentication_invalid(self, reason: str) -> NoReturn:
        """Latch terminal authentication failure and raise typed broker error."""
        safe_reason = str(reason or "authentication_failed")[:256]
        if not self._auth_invalid:
            self._auth_invalid = True
            self._auth_invalid_reason = safe_reason
            self._auth_invalid_at = self._log_time_fn()
            self._auth_failure_generation += 1
            self._clear_rest_caches()
            LOGGER.error(
                "ZERODHA_AUTH_INVALIDATED reason=%s generation=%s",
                safe_reason,
                self._auth_failure_generation,
                extra={
                    "event": "ZERODHA_AUTH_INVALIDATED",
                    "reason": safe_reason,
                    "generation": self._auth_failure_generation,
                },
            )
            callback = self._auth_failure_callback
            if callback is not None:
                with suppress(Exception):
                    callback(self.authentication_status_snapshot())
        raise BrokerAuthenticationError(
            f"Zerodha authentication invalid: {self._auth_invalid_reason}"
        )

    def _raise_if_authentication_latched(self) -> None:
        if not self._auth_invalid:
            return
        now = self._log_time_fn()
        if now >= self._auth_reprobe_next:
            # Let exactly one request through per interval as a re-probe;
            # a success clears the latch via _reset_transient_state.
            self._auth_reprobe_next = now + self._auth_reprobe_interval
            return
        raise BrokerAuthenticationError(
            f"Zerodha authentication invalid: {self._auth_invalid_reason}"
        )

    def quote_api_available(self) -> bool:
        """Return quote API capability flag."""
        return bool(self._quote_api_available)

    def quote_api_status_snapshot(self) -> dict[str, Any]:
        """Return quote API capability status."""
        return {
            "available": bool(self._quote_api_available),
            "error": self._quote_api_error,
            "last_checked_at": self._quote_api_last_checked_at,
        }

    def _is_quote_access_denied(self, exc: Exception) -> bool:
        """Detect terminal access denied state from broker quote calls."""
        text = str(exc).lower()
        return (
            "access denied" in text
            or "forbidden" in text
            or "http 403" in text
            or "403" in text
        )

    def _mark_quote_api_status(
        self, *, available: bool, error: str | None = None
    ) -> None:
        """Record quote API capability state."""
        self._quote_api_available = bool(available)
        self._quote_api_error = str(error) if error else None
        self._quote_api_last_checked_at = time.time()

    def _load_rest_cache(
        self, cache: _RestCacheEntry | None, *, label: str
    ) -> Any | None:
        """Load cached payload if fresh. Args: cache, label. Returns: Any | None. Raises: None."""

        if cache is None:
            return None
        now = self._log_time_fn()
        age = now - cache.updated_at
        if cache.is_fresh(self._rest_cache_ttl, now):
            LOGGER.info(
                "Condition met: zerodha_rest_cache_fallback",
                extra={
                    "event": "zerodha_rest_cache_fallback",
                    "label": label,
                    "age": age,
                },
            )
            return _copy_cache_payload(cache.payload)
        LOGGER.debug(
            "zerodha_rest_cache_stale label=%s age=%0.1fs",
            label,
            age,
            extra={"event": "zerodha_rest_cache_stale", "label": label, "age": age},
        )
        return None

    def quote_any(self, items: Sequence[object]) -> Mapping[str, Any] | None:
        """Fetch Zerodha quote payloads for mixed identifiers.

        Args:
            items: Sequence containing instrument tokens or ``EXCHANGE:SYMBOL``
                identifiers accepted by Zerodha.

        Returns:
            Mapping[str, Any] | None: Quote payloads keyed by identifier when
            available; ``None`` if the request yields no data.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient.quote_any",
            extra={"event": "zerodha_quote_any_enter", "count": len(items)},
        )
        if not items:
            return None

        tokens: list[int] = []
        symbols: list[str] = []
        alias_map: dict[str, str] = {}
        for item in items:
            if isinstance(item, (int, float)):
                try:
                    token_value = int(item)
                except (TypeError, ValueError):
                    continue
                if token_value > 0:
                    tokens.append(token_value)
            elif isinstance(item, str):
                candidate = item.strip().upper()
                if not candidate:
                    continue
                if ":" not in candidate:
                    candidate = f"{self._default_exchange}:{candidate}"
                if candidate == "NSE:NIFTY":
                    candidate = "NSE:NIFTY"
                symbols.append(candidate)
                alias_map.setdefault(candidate, candidate)
                alias_map.setdefault(candidate.split(":", 1)[-1], candidate)

        request_items: list[str] = []
        token_symbol_map: dict[int, str] = {}
        if tokens:
            try:
                token_symbols, symbol_map = self._tokens_to_symbols(tokens)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in ZerodhaKiteClient.quote_any token resolution: %s",
                    exc,
                    extra={"event": "zerodha_quote_any_token_error"},
                    exc_info=exc,
                )
                token_symbols = []
                symbol_map = {}
            if token_symbols:
                request_items.extend(token_symbols)
                for symbol, token in symbol_map.items():
                    token_symbol_map[int(token)] = symbol
                    alias_map.setdefault(symbol.split(":", 1)[-1], symbol)
            else:
                for token in tokens:
                    token_symbol_map[token] = str(token)
                    request_items.append(str(token))

        request_items.extend(symbols)
        if not request_items:
            return None

        ordered_items = list(dict.fromkeys(request_items))
        try:
            self._acquire_bucket(self._QUOTE_BUCKET)
            response = self._ensure_json(
                self._make_request("GET", "/quote", params={"i": ordered_items})
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_quote_access_denied(exc):
                self._mark_quote_api_status(available=False, error="access_denied")
            LOGGER.error(
                "Failure in ZerodhaKiteClient.quote_any request: %s",
                exc,
                extra={"event": "zerodha_quote_any_request_error"},
                exc_info=exc,
            )
            return None

        data = response.get("data")
        self._mark_quote_api_status(available=True)
        if not isinstance(data, Mapping):
            LOGGER.info(
                "Condition met: zerodha_quote_any_empty",
                extra={"event": "zerodha_quote_any_empty"},
            )
            return None

        quotes: dict[str, Any] = {}
        for key, payload in data.items():
            normalized_key = str(key)
            if isinstance(payload, Mapping):
                quotes[normalized_key] = dict(payload)
            else:
                quotes[normalized_key] = payload

        for alias, canonical in alias_map.items():
            if alias in quotes:
                continue
            if canonical in quotes:
                payload = quotes[canonical]
                if isinstance(payload, Mapping):
                    quotes[alias] = dict(payload)
                else:
                    quotes[alias] = payload

        for token, canonical_symbol in token_symbol_map.items():
            token_key = str(token)
            payload = quotes.get(token_key)
            if not isinstance(payload, Mapping):
                payload = quotes.get(canonical_symbol) or quotes.get(
                    canonical_symbol.split(":", 1)[-1]
                )
            if isinstance(payload, Mapping):
                enriched = dict(payload)
                enriched.setdefault("instrument_token", int(token))
                quotes[token_key] = enriched

        return quotes or None

    # BaseBrokerClient protocol methods
    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get quote for symbol.

        Kite's ``/quote`` endpoint keys the response by the exact instrument
        identifier it recognises (e.g. ``NSE:NIFTY 50`` for the index).  A
        caller-facing alias like ``NSE:NIFTY`` is not accepted by the API and
        results in an empty ``data`` payload.  We therefore send every known
        alias for index-style symbols in a single GET and accept whichever
        variant Kite returns.
        """

        self._acquire_bucket(self._QUOTE_BUCKET)
        policy = getattr(self, "_md_policy", MarketDataPolicy.from_env())
        kite_symbol = self._format_symbol(symbol)
        variants = policy.quote_aliases(kite_symbol)

        # Send every alias to Kite so a single round-trip covers any naming
        # mismatch.  Kite silently drops unknown keys from the response.
        try:
            response = self._ensure_json(
                self._make_request("GET", "/quote", params={"i": variants})
            )
        except Exception as exc:
            if self._is_quote_access_denied(exc):
                self._mark_quote_api_status(available=False, error="access_denied")
                throttle_window = max(
                    1.0, float(policy.log_throttle_quote_errors_seconds)
                )
                now = time.monotonic()
                key = f"quote:{str(symbol).upper()}"
                last = self._last_quote_error_log_at.get(key, 0.0)
                if now - last >= throttle_window:
                    self._last_quote_error_log_at[key] = now
                    LOGGER.error(
                        "quote_request_access_denied label=quote symbol=%s error=%s",
                        symbol,
                        exc,
                        extra={
                            "event": "quote_request_access_denied",
                            "label": "quote",
                            "symbol": str(symbol),
                        },
                    )
                else:
                    LOGGER.debug(
                        "quote_request_access_denied_suppressed label=quote symbol=%s",
                        symbol,
                        extra={
                            "event": "quote_request_access_denied_suppressed",
                            "label": "quote",
                            "symbol": str(symbol),
                        },
                    )
            raise
        data = response.get("data", {})
        self._mark_quote_api_status(available=True)
        quote_data: Mapping[str, Any] | None = None
        if isinstance(data, Mapping):
            direct = data.get(kite_symbol)
            if isinstance(direct, Mapping):
                quote_data = direct
            else:
                for key, payload in data.items():
                    if not isinstance(payload, Mapping):
                        continue
                    key_text = str(key).strip().upper()
                    normalized_key = key_text.replace(" ", "").replace(
                        "NIFTY50", "NIFTY"
                    )
                    if key_text in variants:
                        quote_data = payload
                        break
                    for variant in variants:
                        normalized_variant = variant.replace(" ", "").replace(
                            "NIFTY50", "NIFTY"
                        )
                        if key_text == variant or normalized_key == normalized_variant:
                            quote_data = payload
                            break
                    if quote_data is not None:
                        break
                if quote_data is None:
                    # Kite may also return the payload keyed by instrument
                    # token (happens for some newly-listed series).  Any single
                    # payload back is still authoritative for this request.
                    for payload in data.values():
                        if isinstance(payload, Mapping) and payload:
                            quote_data = payload
                            break
        if quote_data is None:  # pragma: no cover - API invariant
            raise BrokerError(f"Quote data missing for {symbol}")

        depth = quote_data.get("depth", {})
        buy_depth = depth.get("buy", [])
        sell_depth = depth.get("sell", [])
        last_trade_time = quote_data.get("last_trade_time")
        ts_ms = int(time.time() * 1000)
        if isinstance(last_trade_time, str):
            try:
                ts_ms = int(datetime.fromisoformat(last_trade_time).timestamp() * 1000)
            except ValueError:
                LOGGER.debug("Unable to parse last_trade_time for %s", symbol)

        return {
            "symbol": symbol,
            "ltp": float(quote_data.get("last_price", 0.0)),
            "ts_ms": ts_ms,
            "bid": float(buy_depth[0]["price"]) if buy_depth else None,
            "ask": float(sell_depth[0]["price"]) if sell_depth else None,
            "volume": quote_data.get("volume"),
            "oi": quote_data.get("oi"),
        }

    def _build_kite_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Construct a sanitized Kite order payload."""

        resolver = getattr(self, "_resolver", None)
        symbol = payload.get("symbol")
        if not symbol:
            raise BrokerError("Missing symbol")
        resolver_exchange: str | None = None
        if symbol and resolver is not None and hasattr(resolver, "exchange_for_symbol"):
            resolver_exchange = cast(Any, resolver).exchange_for_symbol(symbol)

        exchange = (
            resolver_exchange
            or payload.get("exchange")
            or os.getenv("INSTRUMENTS__TRADE_EXCHANGE")
            or "NFO"
        )
        if not exchange:
            raise BrokerError("Missing exchange")

        resolver_symbol: str | None = None
        if (
            symbol
            and resolver is not None
            and hasattr(resolver, "tradingsymbol_for_order")
        ):
            resolver_symbol = cast(Any, resolver).tradingsymbol_for_order(symbol)

        symbol_text = str(symbol).strip()
        if ":" in symbol_text:
            symbol_prefix = symbol_text.split(":", 1)[0].strip().upper()
            if symbol_prefix and symbol_prefix != "NFO":
                raise BrokerError("Exchange must be NFO for NIFTY options")

        tradingsymbol = payload.get("tradingsymbol") or resolver_symbol or symbol
        if not tradingsymbol:
            raise BrokerError("Missing tradingsymbol")

        tradingsymbol_stripped = str(tradingsymbol).strip()
        if ":" in tradingsymbol_stripped:
            tradingsymbol_stripped = tradingsymbol_stripped.split(":", 1)[-1]
        tradingsymbol_stripped = tradingsymbol_stripped.strip()
        tradingsymbol_check = tradingsymbol_stripped.upper()

        if tradingsymbol_check.endswith("FUT"):
            raise BrokerError("Futures disabled for this bot")

        if not tradingsymbol_check.endswith("CE") and not tradingsymbol_check.endswith(
            "PE"
        ):
            raise BrokerError("Only NIFTY options (CE/PE) are allowed")

        exchange_clean = str(exchange).strip().upper()
        if exchange_clean != "NFO":
            raise BrokerError("Only NFO exchange is supported for NIFTY options")

        side = str(payload.get("side") or payload.get("transaction_type") or "").upper()
        if side not in {"BUY", "SELL"}:
            raise BrokerError("Missing side")

        quantity_raw = payload.get("quantity")
        if quantity_raw is None:
            raise BrokerError("Missing quantity")
        try:
            quantity = int(quantity_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError("Invalid quantity") from exc
        if quantity <= 0:
            raise BrokerError("Invalid quantity")

        order_type_value = payload.get("order_type", "MARKET")
        order_type = (
            str(order_type_value.value)
            if hasattr(order_type_value, "value")
            else str(order_type_value)
        ).upper()

        params: dict[str, Any] = {
            "exchange": "NFO",
            "tradingsymbol": tradingsymbol_stripped,
            "transaction_type": side,
            "quantity": quantity,
            "order_type": "MARKET" if order_type == "MARKET" else "LIMIT",
            "product": payload.get("product", "MIS"),
            "validity": payload.get("validity", "DAY"),
        }
        if params["order_type"] == "LIMIT":
            price_value = payload.get("price")
            if price_value is None:
                raise BrokerError("LIMIT order requires price")
            if isinstance(price_value, (int, float, str)):
                params["price"] = float(price_value)
            else:
                raise BrokerError("Invalid price type")

        if "tag" in payload and payload["tag"] is not None:
            params["tag"] = payload["tag"]
        if "parent_order_id" in payload and payload["parent_order_id"] is not None:
            params["parent_order_id"] = payload["parent_order_id"]
        if "disclosed_quantity" in payload and payload["disclosed_quantity"]:
            params["disclosed_quantity"] = int(payload["disclosed_quantity"])

        return params

    def place_order(
        self,
        variety: str = "regular",  # Ensure default is "regular"
        tag: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Place order with Robust Idempotency, Symbol Parsing & Type Mapping.
        """
        import uuid

        # 1. Construct Param Dictionary
        params = kwargs.copy()
        # [FIX] Ensure variety is never empty
        variety = variety or "regular"
        params["variety"] = variety

        # [FIX] Automatic Symbol Resolution
        if "symbol" in params:
            raw_sym = params.pop("symbol")
            if ":" in raw_sym:
                exch, sym = raw_sym.split(":", 1)
                params["exchange"] = exch
                params["tradingsymbol"] = sym
            else:
                params["exchange"] = params.get("exchange", "NFO")
                params["tradingsymbol"] = raw_sym

        # [FIX] Map 'side' to Kite Transaction Type
        if "side" in params:
            side_val = str(params.pop("side")).upper()
            params["transaction_type"] = "BUY" if "BUY" in side_val else "SELL"

        # [FIX] Robust Order Type Mapping
        if "order_type" in params:
            raw_ot = params["order_type"]
            ot_str = getattr(raw_ot, "value", str(raw_ot)).upper()
            mapping = {
                "STOP_LOSS_MARKET": "SL-M",
                "STOP_LOSS_LIMIT": "SL",
                "STOP_LOSS": "SL",
                "MARKET": "MARKET",
                "LIMIT": "LIMIT",
                "SL": "SL",
                "SL-M": "SL-M",
            }
            params["order_type"] = mapping.get(ot_str, ot_str)

        # 2. Generate Safe Unique Tag (Idempotency Key)
        # [FIX] Use client_order_id if provided by OrderManager to ensure Retries don't duplicate
        client_id = kwargs.get("client_order_id") or kwargs.get("tag")

        if client_id:
            # Use the deterministic ID provided by the caller
            final_tag = str(client_id)[:20]  # Zerodha tag limit is 20 chars
        else:
            # Fallback to random if not provided (Legacy behavior)
            unique_id = uuid.uuid4().hex[:8]
            raw_tag = str(tag or "bot").strip()
            safe_prefix = raw_tag[:11]
            final_tag = f"{safe_prefix}_{unique_id}"

        params["tag"] = final_tag

        if hasattr(self, "_acquire_bucket") and hasattr(self, "_ORDER_BUCKET"):
            self._acquire_bucket(self._ORDER_BUCKET)

        # [FIX] Filter out None values and normalize Kite-specific parameters once.
        clean_params = self._normalize_order_params(params)

        try:
            # 3. Attempt Placement
            # [FIX] Ensure method is explicitly "POST"
            response = self._ensure_json(
                self._make_request(
                    "POST",
                    f"/orders/{variety}",  # e.g. /orders/regular
                    data=clean_params,
                    expect_order_response=True,
                    operation_label="orders.place",
                )
            )

            data = response.get("data", {})
            order_id = data.get("order_id")
            if not order_id:
                raise OrderPlacementError(
                    "Order placement acknowledged without order_id"
                )
            return {
                "order_id": order_id,
                "submitted": True,
                "status": "SUBMITTED",
                "raw_status": response.get("status", "success"),
                "message": response.get("message", ""),
                "tag": final_tag,
                "raw_response": response,
            }

        except Exception as e:
            # [FIX] Add specific logging for 405 errors
            if "405" in str(e):
                self._logger.critical(
                    f"🛑 Zerodha 405 Error (Bad URL/Method). URL: /orders/{variety}, Method: POST"
                )

            # [FIX] Fail Fast Logic
            raise OrderPlacementError(f"Order placement failed: {e}")

    def _normalize_order_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return Kite-compatible order parameters for a single broker call."""

        clean_params = {k: v for k, v in params.items() if v is not None}
        order_type = str(clean_params.get("order_type", "")).upper()
        if order_type not in {"MARKET", "LIMIT", "SL", "SL-M"}:
            raise BrokerError(
                f"Unsupported Zerodha order_type: {order_type or 'missing'}"
            )
        clean_params["order_type"] = order_type
        if order_type == "MARKET":
            clean_params.pop("trigger_price", None)
            clean_params.pop("price", None)
        if order_type in {"MARKET", "SL-M"}:
            clean_params["market_protection"] = self._normalize_market_protection(
                clean_params.get(
                    "market_protection",
                    os.getenv("ZERODHA_MARKET_PROTECTION", "-1"),
                )
            )
        else:
            clean_params.pop("market_protection", None)
        return clean_params

    @staticmethod
    def _normalize_market_protection(raw_value: Any) -> int:
        """Validate Zerodha market protection: -1 or integer 1..100."""

        try:
            protection_float = float(raw_value)
            protection = int(protection_float)
        except (TypeError, ValueError) as exc:
            raise BrokerError(
                "market_protection must be -1 or an integer from 1 to 100"
            ) from exc
        if protection_float != float(protection):
            raise BrokerError(
                "market_protection must be -1 or an integer from 1 to 100"
            )
        if protection != -1 and not 1 <= protection <= 100:
            raise BrokerError(
                "market_protection must be -1 or an integer from 1 to 100"
            )
        return protection

    # Additional Kite-specific methods
    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Get last traded price for multiple symbols."""

        if not symbols:
            return {}
        self._acquire_bucket(self._QUOTE_BUCKET)
        kite_symbols = [self._format_symbol(symbol) for symbol in symbols]
        response = self._ensure_json(
            self._make_request("GET", "/quote/ltp", params={"i": kite_symbols})
        )
        data = cast(dict[str, Any], response.get("data", {}))
        results: dict[str, float] = {}
        for symbol in symbols:
            symbol_key = self._format_symbol(symbol)
            ltp_entry = data.get(symbol_key)
            if not ltp_entry:
                continue
            results[symbol] = float(ltp_entry.get("last_price", 0.0))
        return results

    def attach_resolver(self, resolver: InstrumentResolver) -> None:
        """Attach an instrument resolver for token-symbol translation."""

        self._resolver = resolver

    def get_ltp_bulk(self, tokens: list[int]) -> dict[int, float]:
        """Return mapping of instrument tokens to last traded price."""

        if not tokens:
            return {}
        policy = getattr(self, "_md_policy", MarketDataPolicy.from_env())

        # When POLL_REQUIRE_DEPTH is enabled we prefer the /quote endpoint (depth)
        # so that polling mode retrieves full quote payloads (bid/ask/depth) rather
        # than LTP-only. This helps decision gates that rely on spread/orderbook.
        require_depth = os.getenv("POLL_REQUIRE_DEPTH", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if require_depth:
            try:
                # get_quote_bulk already honors rate limiting and symbol resolution.
                quote_map = self.get_quote_bulk(tokens)
                out: dict[int, float] = {}
                for symbol, payload in quote_map.items():
                    if not isinstance(payload, Mapping):
                        continue
                    try:
                        token = int(
                            payload.get("instrument_token")
                            or payload.get("instrument_token_id")
                            or 0
                        )
                    except (TypeError, ValueError):
                        token = 0
                    if token <= 0:
                        continue
                    try:
                        last_price = float(payload.get("last_price", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if last_price > 0:
                        out[token] = last_price

                if out:
                    now = time.time()
                    if now - self._last_log_ltp_bulk >= self._log_throttle_interval:
                        LOGGER.info(
                            "zerodha_get_ltp_bulk_using_depth",
                            extra={
                                "event": "zerodha_get_ltp_bulk_depth_used",
                                "count": len(out),
                            },
                        )
                        self._last_log_ltp_bulk = now
                    return out

                LOGGER.info(
                    "zerodha_get_ltp_bulk_depth_empty_fallback",
                    extra={
                        "event": "zerodha_get_ltp_bulk_depth_empty_fallback",
                        "tokens": tokens,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - fall back to ltp endpoint
                LOGGER.warning(
                    "Depth fetch failed, falling back to LTP-only: %s",
                    exc,
                    extra={"event": "zerodha_get_ltp_bulk_depth_error"},
                )

        # Fallback to the LTP-only endpoint
        canonical_input = all(
            isinstance(item, str) and ":" in str(item).strip() for item in tokens
        )
        if canonical_input:
            symbols = [str(item).strip() for item in tokens]
            symbol_map: dict[str, int] = {}
        else:
            symbols, symbol_map = self._tokens_to_symbols(tokens)
        if not symbols:
            return {}
        self._acquire_bucket(self._QUOTE_BUCKET)
        response = self._ensure_json(
            self._make_request("GET", "/quote/ltp", params={"i": symbols})
        )
        data = cast(dict[str, Any], response.get("data", {}))
        out: dict[int, float] = {}
        for symbol, payload in data.items():
            token = int(payload.get("instrument_token") or symbol_map.get(symbol) or 0)
            if token <= 0:
                continue
            try:
                last_price = float(payload.get("last_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if last_price > 0:
                out[token] = last_price

        now = time.time()
        if now - self._last_log_ltp_bulk >= self._log_throttle_interval:
            LOGGER.info(
                "zerodha_get_ltp_bulk_ltp_fallback",
                extra={
                    "event": "zerodha_get_ltp_bulk_ltp_fallback",
                    "count": len(out),
                },
            )
            self._last_log_ltp_bulk = now
        return out

    def get_quote_by_token(self, token: int) -> dict[str, Any]:
        """Return a Zerodha quote payload for the given instrument token."""

        symbols, _symbol_map = self._tokens_to_symbols([int(token)])
        if not symbols:
            msg = f"Instrument resolver missing for token {token}"
            raise BrokerError(msg)
        self._acquire_bucket(self._QUOTE_BUCKET)
        response = self._ensure_json(
            self._make_request("GET", "/quote", params={"i": symbols})
        )
        data = cast(dict[str, Any], response.get("data", {}))
        payload = dict(data.get(symbols[0], {}))
        payload["instrument_token"] = int(token)
        if "last_price" not in payload:
            payload["last_price"] = 0.0
        return payload

    def get_quote_bulk(
        self, tokens: list[int] | list[str]
    ) -> dict[str, dict[str, Any]]:
        """Return mapping of symbols to Zerodha quote payloads.

        ✅ PRODUCTION FIX: Now accepts both integer tokens AND symbol strings.
        Returns dict keyed by symbol string for consistency with Zerodha API response.

        Args:
            tokens: List of integer tokens OR symbol strings (e.g., ["NSE:NIFTY"])

        Returns:
            Dict mapping symbol strings to quote payloads
        """
        if not tokens:
            return {}

        policy = getattr(self, "_md_policy", MarketDataPolicy.from_env())

        canonical_input = all(
            isinstance(item, str) and ":" in str(item).strip() for item in tokens
        )
        # alias_to_original maps every alias variant we send to Kite back to
        # the caller's requested symbol so the response can be re-keyed in the
        # caller's preferred form (e.g. "NSE:NIFTY" instead of "NSE:NIFTY 50").
        alias_to_original: dict[str, str] = {}
        original_symbols: list[str] = []
        if canonical_input:
            original_symbols = [str(item).strip() for item in tokens]
            symbols = []
            for original in original_symbols:
                aliases = policy.quote_aliases(original)
                # Always include the original even when not in the alias list.
                if original and original not in aliases:
                    aliases.insert(0, original)
                for alias in aliases:
                    if alias and alias not in symbols:
                        symbols.append(alias)
                    alias_upper = alias.strip().upper()
                    alias_to_original.setdefault(alias_upper, original)
            symbol_map: dict[str, int] = {}
            LOGGER.debug(
                "quote_bulk canonical fast-path",
                extra={
                    "event": "quote_bulk_canonical_fast_path",
                    "symbols": symbols[:8],
                    "count": len(symbols),
                    "originals": original_symbols[:8],
                },
            )
        else:
            symbols, symbol_map = self._tokens_to_symbols(tokens)
        if not symbols:
            LOGGER.warning(
                "Token-to-symbol mapping empty",
                extra={
                    "event": "quote_bulk_mapping_empty",
                    "tokens": str(tokens)[:100],
                },
            )
            return {}

        self._acquire_bucket(self._QUOTE_BUCKET)
        try:
            response = self._ensure_json(
                self._make_request("GET", "/quote", params={"i": symbols})
            )
        except Exception as exc:
            if self._is_quote_access_denied(exc):
                self._mark_quote_api_status(available=False, error="access_denied")
                # 403s repeat every poll interval; demote to debug after the
                # first marker so logs do not flood.
                LOGGER.debug(
                    "Quote bulk denied (access_denied): %s",
                    exc,
                    extra={
                        "event": "quote_bulk_access_denied",
                        "symbols_count": len(symbols),
                    },
                )
            elif isinstance(exc, BrokerAuthenticationError):
                # Terminal auth failures repeat at poll cadence (233 ERROR
                # lines in 4 minutes on 2026-07-07); throttle to one per
                # minute — ZERODHA_AUTH_INVALIDATED already fired loudly once.
                log_throttled(
                    LOGGER,
                    "quote_bulk_auth_invalid",
                    "Quote bulk blocked by invalid auth (throttled): %s" % exc,
                    interval_sec=60,
                    level=logging.ERROR,
                    extra={
                        "event": "quote_bulk_request_error",
                        "symbols_count": len(symbols),
                    },
                )
            else:
                LOGGER.error(
                    "Quote bulk request failed: %s",
                    exc,
                    extra={
                        "event": "quote_bulk_request_error",
                        "symbols_count": len(symbols),
                    },
                )
            return {}

        data = cast(dict[str, Any], response.get("data", {}))
        self._mark_quote_api_status(available=True)

        # ✅ FIX: Return keyed by the caller's requested symbol when possible
        # (so callers asking for "NSE:NIFTY" find the payload there even though
        # Kite returned it under "NSE:NIFTY 50").
        out: dict[str, dict[str, Any]] = {}
        for returned_symbol, payload in data.items():
            if not payload:
                continue
            payload_dict = dict(payload)
            if "instrument_token" not in payload_dict:
                token_from_map = symbol_map.get(returned_symbol)
                if token_from_map:
                    payload_dict["instrument_token"] = token_from_map
            original_key = alias_to_original.get(
                str(returned_symbol).strip().upper(), returned_symbol
            )
            out.setdefault(original_key, payload_dict)
            # Keep Kite's exact response key as a fallback so callers that
            # pre-translate to "NSE:NIFTY 50" themselves still find it.
            if returned_symbol not in out:
                out[returned_symbol] = dict(payload_dict)

        return out

    def quote(self, instruments: list[str] | list[int] | str | int) -> dict[str, Any]:
        """Standard KiteConnect compliant alias for quote fetching.

        ✅ PRODUCTION FIX: Now accepts both symbol strings AND integer tokens.

        Args:
            instruments: List of symbols (e.g., ["NSE:NIFTY"]) OR tokens OR single value

        Returns:
            Dict mapping symbol strings to quote payloads
        """
        # Handle single input
        if isinstance(instruments, (str, int)):
            instruments = [instruments]
        return self.get_quote_bulk(instruments)

    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """Get historical OHLC data."""

        self._acquire_bucket(self._HISTORICAL_BUCKET)
        instrument_token = self.get_instrument_token(symbol)
        response = self._ensure_json(
            self._make_request(
                "GET",
                f"/instruments/historical/{instrument_token}/{interval}",
                params={"from": from_date, "to": to_date},
            )
        )
        data = cast(dict[str, Any], response.get("data", {}))
        return cast(list[dict], data.get("candles", []))

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime | str,
        to_date: datetime | str,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        """
        KiteConnect-compatible historical data fetcher.
        Required by MarketDataManager/Runner backfill logic.
        """
        token = _normalize_historical_token(instrument_token)
        from_value = _format_kite_datetime(from_date)
        to_value = _format_kite_datetime(to_date)

        # 2. Build Parameters
        params = {
            "from": from_value,
            "to": to_value,
            "continuous": 1 if continuous else 0,
            "oi": 1 if oi else 0,
        }

        # 3. Execute Request (Using your native infrastructure)
        self._acquire_bucket(self._HISTORICAL_BUCKET)
        response = self._ensure_json(
            self._make_request(
                "GET",
                f"/instruments/historical/{token}/{interval}",
                params=params,
            )
        )

        # 4. Return standard list of candles
        data = cast(dict[str, Any], response.get("data", {}))
        return cast(list[dict], data.get("candles", []))

    def get_order_status(self, order_id: str) -> dict:
        """Get order status from broker."""

        self._acquire_bucket(self._GENERAL_BUCKET)
        response = self._ensure_json(self._make_request("GET", f"/orders/{order_id}"))
        orders = cast(list[dict], response.get("data", []))
        for order in reversed(orders):
            if str(order.get("order_id")) == str(order_id):
                return order
        return {}

    def cancel_order(self, order_id: str, variety: str = "regular") -> dict:
        """Cancel order."""

        self._acquire_bucket(self._ORDER_BUCKET)
        response = self._ensure_json(
            self._make_request(
                "DELETE",
                f"/orders/{variety}/{order_id}",
                operation_label="orders.cancel",
            )
        )
        return cast(dict[str, Any], response.get("data", {}))

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        variety: str = "regular",
    ) -> dict:
        """Modify order."""

        if quantity is None and price is None:
            raise OrderPlacementError("Must provide quantity or price to modify order")

        payload: dict[str, Any] = {}
        if quantity is not None:
            payload["quantity"] = quantity
        if price is not None:
            payload["price"] = float(price)
            payload["trigger_price"] = float(price)

        self._acquire_bucket(self._ORDER_BUCKET)
        response = self._ensure_json(
            self._make_request(
                "PUT",
                f"/orders/{variety}/{order_id}",
                data=payload,
                operation_label="orders.modify",
            )
        )
        return cast(dict[str, Any], response.get("data", {}))

    def get_orders(self) -> list[dict]:
        """Get all Zerodha orders for the trading day (Log-Silent if empty)."""
        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_orders",
            extra={"event": "zerodha_get_orders_start"},
        )
        label = "orders.fetch"
        should_retry, on_retry = self._build_retry_handlers(endpoint="/orders")

        def _operation() -> list[dict]:
            with _BROKER_SYNC_LOCK:
                self._acquire_bucket(self._GENERAL_BUCKET)
                payload = self._ensure_json(
                    self._make_request(
                        "GET", "/orders", operation_label=label, with_retry=False
                    )
                )
                orders = cast(list[dict], payload.get("data", []))

            # [FIX] Only log INFO if we actually have orders, otherwise DEBUG
            if orders:
                LOGGER.info(
                    "zerodha_orders_fetch_success count=%d",
                    len(orders),
                    extra={
                        "event": "zerodha_orders_fetch_success",
                        "count": len(orders),
                    },
                )
            else:
                LOGGER.debug("zerodha_orders_fetch_success count=0")
            self._orders_cache = _RestCacheEntry(
                payload=list(orders),
                updated_at=self._log_time_fn(),
            )
            return orders

        try:
            return self._execute_with_retry(
                label=label,
                operation=_operation,
                should_retry=should_retry,
                error_message="Failed to fetch Zerodha orders",
                on_retry=on_retry,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_orders: %s",
                exc,
                extra={"event": "zerodha_get_orders_error"},
            )
            cached = self._load_rest_cache(self._orders_cache, label=label)
            if cached is not None:
                return cast(list[dict], cached)
            raise

    def get_positions(self) -> list[dict[str, Any]]:
        """Return one authoritative Zerodha net-position snapshot.

        Position reconciliation never falls back to cache: a failed or malformed
        broker response is unknown exposure, not confirmed flatness.
        """
        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_positions",
            extra={"event": "zerodha_get_positions_start"},
        )
        label = "positions.fetch"
        endpoint = "/portfolio/positions"
        should_retry, on_retry = self._build_retry_handlers(endpoint=endpoint)

        def _operation() -> list[dict[str, Any]]:
            with _BROKER_SYNC_LOCK:
                self._acquire_bucket(self._GENERAL_BUCKET)
                response = self._ensure_json(
                    self._make_request(
                        "GET", endpoint, operation_label=label, with_retry=False
                    )
                )
            payload = response.get("data")
            if not isinstance(payload, Mapping):
                raise BrokerError("Malformed positions response: data object missing")
            net_positions = payload.get("net")
            if not isinstance(net_positions, list):
                raise BrokerError("Malformed positions response: data.net list missing")

            normalized: list[dict[str, Any]] = []
            for index, row in enumerate(net_positions):
                if not isinstance(row, Mapping):
                    raise BrokerError(
                        f"Malformed positions response: data.net[{index}] is not an object"
                    )
                normalized.append(dict(row))

            if normalized:
                LOGGER.info(
                    "zerodha_positions_fetch_success count=%d",
                    len(normalized),
                    extra={
                        "event": "zerodha_positions_fetch_success",
                        "count": len(normalized),
                    },
                )
            else:
                LOGGER.debug("zerodha_positions_fetch_success count=0")
            self._positions_cache = _RestCacheEntry(
                payload=list(normalized),
                updated_at=self._log_time_fn(),
            )
            return normalized

        try:
            return self._execute_with_retry(
                label=label,
                operation=_operation,
                should_retry=should_retry,
                error_message="Failed to fetch Zerodha positions",
                on_retry=on_retry,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_positions: %s",
                exc,
                extra={"event": "zerodha_get_positions_error"},
                exc_info=exc,
            )
            raise

    def get_holdings(self) -> list[dict]:
        """Get holdings."""

        self._acquire_bucket(self._GENERAL_BUCKET)
        response = self._ensure_json(
            self._make_request(
                "GET", "/portfolio/holdings", operation_label="holdings.fetch"
            )
        )
        return cast(list[dict], response.get("data", []))

    def get_account_margins(self, segment: str | None = None) -> dict[str, Any]:
        """Return raw Zerodha account margin payload for *segment*.

        Args:
            segment: Zerodha segment identifier such as ``"equity"``.

        Returns:
            dict[str, Any]: Raw margin payload returned by the broker API.

        Raises:
            BrokerError: If the broker API call ultimately fails.
        """

        normalized_segment = (
            str(segment or self._default_margin_segment).strip().lower()
            or self._default_margin_segment
        )
        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_account_margins",
            extra={
                "event": "zerodha_get_account_margins_start",
                "segment": normalized_segment,
            },
        )
        label = f"user_margins.{normalized_segment}"
        endpoint = f"/user/margins/{normalized_segment}"
        should_retry, on_retry = self._build_retry_handlers(endpoint=endpoint)

        def _operation() -> dict[str, Any]:
            self._acquire_bucket(self._GENERAL_BUCKET)
            raw_payload = self._ensure_json(
                self._make_request(
                    "GET",
                    endpoint,
                    operation_label=label,
                    with_retry=False,
                )
            )
            payload: dict[str, Any] = {}
            if isinstance(raw_payload, Mapping):
                data_section = raw_payload.get("data")
                if isinstance(data_section, Mapping):
                    payload = dict(data_section)
                else:
                    payload = dict(cast(Mapping[str, Any], raw_payload))
            LOGGER.debug(
                "zerodha_account_margins_fetch_success segment=%s",
                normalized_segment,
                extra={
                    "event": "zerodha_account_margins_fetch_success",
                    "segment": normalized_segment,
                    "keys": sorted(payload.keys()),
                },
            )
            self._margins_cache[normalized_segment] = _RestCacheEntry(
                payload=dict(payload),
                updated_at=self._log_time_fn(),
            )
            return payload

        try:
            return self._execute_with_retry(
                label=label,
                operation=_operation,
                should_retry=should_retry,
                error_message="Failed to fetch Zerodha account margins",
                on_retry=on_retry,
            )
        except BrokerAuthenticationError:
            raise
        except Exception as exc:  # noqa: BLE001 - propagate after logging
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_account_margins: %s",
                exc,
                extra={
                    "event": "zerodha_get_account_margins_error",
                    "segment": normalized_segment,
                },
                exc_info=exc,
            )
            cached = self._load_rest_cache(
                self._margins_cache.get(normalized_segment),
                label=label,
            )
            if cached is not None:
                return cast(dict[str, Any], cached)
            raise

    def get_margins(self, segment: str = "equity") -> dict[str, Any]:
        """Fetch Zerodha margin information for *segment*.

        Args:
            segment: Zerodha margin segment identifier.

        Returns:
            dict[str, Any]: Margin payload returned by the Zerodha API.

        Raises:
            BrokerError: If the broker API call ultimately fails.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_margins",
            extra={"event": "zerodha_get_margins_start", "segment": segment},
        )
        label = f"margins.{segment}"
        endpoint = f"/margins/{segment}"
        should_retry, on_retry = self._build_retry_handlers(endpoint=endpoint)

        def _operation() -> dict[str, Any]:
            self._acquire_bucket(self._GENERAL_BUCKET)
            raw_payload = self._ensure_json(
                self._make_request(
                    "GET",
                    endpoint,
                    operation_label=label,
                    with_retry=False,
                )
            )
            payload = self._resolve_margin_payload(
                cast(Mapping[str, Any] | None, raw_payload),
                segment=segment,
            )

            # [CORRECTED] Aligned correctly
            now = time.time()
            if now - self._last_log_margins >= self._log_throttle_interval:
                LOGGER.info(
                    "zerodha_margins_fetch_success segment=%s",
                    segment,
                    extra={
                        "event": "zerodha_margins_fetch_success",
                        "segment": segment,
                        "keys": sorted(payload.keys()),
                    },
                )
                self._last_log_margins = now
            return payload

        try:
            return self._execute_with_retry(
                label=label,
                operation=_operation,
                should_retry=should_retry,
                error_message="Failed to fetch Zerodha margins",
                on_retry=on_retry,
            )
        except Exception as exc:  # noqa: BLE001 - propagate after logging
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_margins: %s",
                exc,
                extra={
                    "event": "zerodha_get_margins_error",
                    "segment": segment,
                },
            )
            raise

    def get_margin_summary(self, segment: str = "equity") -> dict[str, float]:
        """Return normalized margin snapshot for a Zerodha segment.

        Args:
            segment: Zerodha segment identifier such as ``"equity"``.

        Returns:
            dict[str, float]: Margin summary with ``available``, ``used``, and ``net``.

        Raises:
            BrokerError: Propagated when the margin fetch fails.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_margin_summary",
            extra={"event": "zerodha_margin_summary_start", "segment": segment},
        )
        try:
            payload = self.get_margins(segment=segment)
        except Exception as exc:  # noqa: BLE001 - propagate after logging
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_margin_summary: %s",
                exc,
                extra={
                    "event": "zerodha_margin_summary_error",
                    "segment": segment,
                },
                exc_info=exc,
            )
            raise

        summary = self._normalize_margin_payload(payload, segment=segment)

        # [CORRECTED] Aligned correctly
        now = time.time()
        if now - self._last_log_margins >= self._log_throttle_interval:
            LOGGER.info(
                "zerodha_margin_summary_success",
                extra={
                    "event": "zerodha_margin_summary_success",
                    "segment": segment,
                    "available": summary.get("available"),
                    "used": summary.get("used"),
                    "net": summary.get("net"),
                },
            )
            self._last_log_margins = now
        return summary

    def get_available_balance(self, segment: str = "equity") -> float:
        """Return broker-reported available account balance for a Zerodha segment.

        Live account funds are fail-closed: only ``/user/margins/{segment}`` is
        accepted, terminal authentication failures are re-raised, and missing or
        malformed broker payloads raise ``BrokerBalanceUnavailableError``.
        """

        self._raise_if_authentication_latched()
        normalized_segment = (
            str(segment or self._default_margin_segment).strip().lower()
            or self._default_margin_segment
        )
        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_available_balance",
            extra={
                "event": "zerodha_available_balance_start",
                "segment": normalized_segment,
            },
        )
        try:
            account_payload = self.get_account_margins(segment=normalized_segment)
            summary = self._parse_account_margin_summary(
                account_payload,
                segment=normalized_segment,
            )
        except BrokerAuthenticationError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert to typed fail-closed error
            LOGGER.error(
                "ZERODHA_BALANCE_REFRESH_FAILED segment=%s reason=%s",
                normalized_segment,
                str(exc),
                extra={
                    "event": "ZERODHA_BALANCE_REFRESH_FAILED",
                    "segment": normalized_segment,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )
            raise BrokerBalanceUnavailableError(
                f"Broker balance unavailable for segment {normalized_segment}: {exc}"
            ) from exc

        available = float(summary["available"])
        snapshot = {
            "available_cash": round(available, 2),
            "live_balance": round(float(summary["live_balance"]), 2),
            "opening_balance": round(float(summary["opening_balance"]), 2),
            "net": round(float(summary["net"]), 2),
        }
        now = time.time()
        snapshot_tuple = (
            snapshot["available_cash"],
            snapshot["live_balance"],
            snapshot["net"],
        )
        heartbeat_interval = float(os.getenv("BALANCE_SUCCESS_LOG_INTERVAL_SECONDS", "900"))
        should_log = (
            not getattr(self, "_balance_success_logged_once", False)
            or snapshot_tuple != getattr(self, "_last_balance_success_snapshot", None)
            or now - getattr(self, "_last_balance_success_log_ts", 0.0) >= heartbeat_interval
        )
        if should_log:
            self._balance_success_logged_once = True
            self._last_balance_success_log_ts = now
            self._last_balance_success_snapshot = snapshot_tuple
            LOGGER.info(
                "ZERODHA_BALANCE_REFRESH_SUCCESS available_cash=%.2f live_balance=%.2f opening_balance=%.2f net=%.2f",
                snapshot["available_cash"],
                snapshot["live_balance"],
                snapshot["opening_balance"],
                snapshot["net"],
                extra={"event": "ZERODHA_BALANCE_REFRESH_SUCCESS", "segment": normalized_segment, **snapshot},
            )
        else:
            LOGGER.debug(
                "ZERODHA_BALANCE_REFRESH_UNCHANGED available_cash=%.2f live_balance=%.2f net=%.2f",
                snapshot["available_cash"],
                snapshot["live_balance"],
                snapshot["net"],
                extra={
                    "event": "ZERODHA_BALANCE_REFRESH_UNCHANGED",
                    "segment": normalized_segment,
                    **snapshot,
                },
            )
        self._last_log_balance = now
        self._last_balance_snapshot = snapshot
        self._last_balance_snapshot_at = now
        return available

    def _parse_account_margin_summary(
        self,
        payload: Mapping[str, Any],
        *,
        segment: str,
    ) -> dict[str, float]:
        """Strictly parse ``/user/margins/{segment}`` account-funds payload."""
        if not isinstance(payload, Mapping) or not payload:
            raise BrokerBalanceUnavailableError("empty_account_margin_payload")
        if self._is_authentication_failure(status_code=None, payload=payload, error_text=""):
            self._mark_authentication_invalid(str(payload.get("message") or payload.get("error_type") or "authentication_failed"))
        if str(payload.get("status", "")).lower() == "error" or "error_type" in payload:
            raise BrokerBalanceUnavailableError("broker_error_margin_payload")

        segment_key = str(segment or "equity").strip().lower() or "equity"
        target: Mapping[str, Any]
        candidate = payload.get(segment_key)
        if isinstance(candidate, Mapping):
            target = candidate
        elif isinstance(payload.get("available"), Mapping) or "net" in payload:
            target = payload
        else:
            raise BrokerBalanceUnavailableError("missing_segment_margin_payload")

        available_map = target.get("available")
        utilised_map = target.get("utilised")
        if not isinstance(available_map, Mapping):
            raise BrokerBalanceUnavailableError("missing_available_margin_fields")
        if not isinstance(utilised_map, Mapping):
            utilised_map = {}

        def _number(name: str, value: Any, *, required: bool = True, allow_negative: bool = False) -> float:
            if value is None:
                if required:
                    raise BrokerBalanceUnavailableError(f"missing_margin_field:{name}")
                return 0.0
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise BrokerBalanceUnavailableError(f"invalid_margin_field:{name}") from exc
            if not math.isfinite(parsed):
                raise BrokerBalanceUnavailableError(f"non_finite_margin_field:{name}")
            if parsed < 0.0 and not allow_negative:
                raise BrokerBalanceUnavailableError(f"negative_margin_field:{name}")
            return parsed

        cash_value = available_map.get("cash")
        if cash_value is None:
            cash_value = available_map.get("live_balance")
        if cash_value is None:
            cash_value = available_map.get("opening_balance")
        available = _number("available.cash", cash_value)
        live_balance = _number("available.live_balance", available_map.get("live_balance"), required=False)
        opening_balance = _number("available.opening_balance", available_map.get("opening_balance"), required=False)
        # Zerodha frequently reports available.cash=0 while the real deployable
        # intraday margin sits in live_balance (and net). Keying off cash alone made
        # the bot see a ₹0 balance (dashboard 'BALANCE —') and refuse to size/trade
        # despite funded margin. When cash is non-positive but live_balance is
        # positive, prefer live_balance as the usable available balance.
        if available <= 0.0 and live_balance > 0.0:
            available = live_balance
        used = 0.0
        for key in ("debits", "span", "exposure", "option_premium", "holding_sales"):
            value = utilised_map.get(key)
            if value is not None:
                # Utilised components can legitimately be negative (e.g. a debits
                # reversal/credit). A negative utilised field must NOT fail the whole
                # balance refresh (which previously cascaded to BROKER NOT READY).
                used += _number(f"utilised.{key}", value, required=False, allow_negative=True)
        net_value = target.get("net")
        net = _number("net", net_value) if net_value is not None else available + used
        return {
            "available": available,
            "used": used,
            "net": net,
            "live_balance": live_balance,
            "opening_balance": opening_balance,
        }

    def _resolve_simulated_balance(self) -> float:
        """Return explicit non-live simulation capital; never used as live fallback."""
        for name in ("RISK__CAPITAL", "RISK_CAPITAL", "BACKTEST__CAPITAL"):
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                continue
            value = float(raw)
            if math.isfinite(value) and value >= 0.0:
                return value
        raise ConfigurationError("simulated_balance_requires_explicit_capital")

    def _resolve_margin_payload(
        self,
        payload: Mapping[str, Any] | None,
        *,
        segment: str,
    ) -> dict[str, Any]:
        """Return a margin payload mapping for Zerodha margin responses.

        Args:
            payload: Raw JSON mapping returned by the margins endpoint.
            segment: Zerodha margin segment identifier.

        Returns:
            dict[str, Any]: Mapping normalized to the requested segment.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient._resolve_margin_payload",
            extra={
                "event": "zerodha_resolve_margin_payload_start",
                "segment": segment,
            },
        )
        normalized_segment = str(segment or "equity").strip().lower() or "equity"
        result: dict[str, Any] = {}
        try:
            if not isinstance(payload, Mapping):
                LOGGER.debug(
                    "Condition met: zerodha_margin_payload_not_mapping",
                    extra={"event": "zerodha_margin_payload_not_mapping"},
                )
                return result

            data_section = payload.get("data")
            if isinstance(data_section, Mapping) and data_section:
                result = dict(data_section)
                LOGGER.info(
                    "Condition met: zerodha_margin_payload_data_section",
                    extra={
                        "event": "zerodha_margin_payload_data_section",
                        "segment": normalized_segment,
                        "keys": sorted(result.keys()),
                    },
                )
                return result

            if isinstance(data_section, Iterable) and not isinstance(
                data_section, (str, bytes)
            ):
                for item in data_section:
                    if not isinstance(item, Mapping):
                        continue
                    raw_segment = str(item.get("segment", "")).strip().lower()
                    if raw_segment == normalized_segment:
                        result = dict(item)
                        LOGGER.info(
                            "Condition met: zerodha_margin_payload_iterable_segment",
                            extra={
                                "event": "zerodha_margin_payload_iterable_segment",
                                "segment": normalized_segment,
                                "keys": sorted(result.keys()),
                            },
                        )
                        return result

            nested_segment = payload.get(normalized_segment)
            if isinstance(nested_segment, Mapping):
                result = dict(nested_segment)
                LOGGER.info(
                    "Condition met: zerodha_margin_payload_nested_segment",
                    extra={
                        "event": "zerodha_margin_payload_nested_segment",
                        "segment": normalized_segment,
                        "keys": sorted(result.keys()),
                    },
                )
                return result

            if result:
                return result

            result = dict(payload)
            LOGGER.info(
                "Condition met: zerodha_margin_payload_fallback",
                extra={
                    "event": "zerodha_margin_payload_fallback",
                    "segment": normalized_segment,
                    "keys": sorted(result.keys()),
                },
            )
            return result

        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ZerodhaKiteClient._resolve_margin_payload: %s",
                exc,
                extra={
                    "event": "zerodha_margin_payload_error",
                    "segment": normalized_segment,
                },
                exc_info=exc,
            )
            return {}

    def _normalize_margin_payload(
        self, payload: Any, *, segment: str
    ) -> dict[str, float]:
        """Normalize Zerodha margin payload into balance fields.

        Args:
            payload: Raw payload returned by :meth:`get_margins`.
            segment: Zerodha segment identifier used for selection.

        Returns:
            dict[str, float]: Mapping containing ``available``, ``used``, and ``net``.

        Raises:
            None.
        """

        segment_key = str(segment or "equity").strip().lower() or "equity"
        target: Mapping[str, Any] | None = None
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(key, str) and key.lower() == segment_key:
                    if isinstance(value, Mapping):
                        target = value
                        break
            if target is None:
                target = payload
        elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                raw_segment = str(item.get("segment", "")).strip().lower()
                if raw_segment == segment_key:
                    target = item
                    break
        if target is None:
            return {"available": 0.0, "used": 0.0, "net": 0.0}

        available_source = target.get("available")
        available_value = self._coerce_positive_float(available_source)
        if isinstance(available_source, Mapping):
            available_mapping = cast(Mapping[str, Any], available_source)
            for key in ("live_balance", "cash", "available_cash", "opening_balance"):
                candidate = self._coerce_positive_float(available_mapping.get(key))
                if candidate is not None:
                    available_value = candidate
                    break
        if available_value is None:
            for key in ("available_cash", "cash", "live_balance"):
                nested_available = self._coerce_positive_float(target.get(key))
                if nested_available is not None:
                    available_value = nested_available
                    break

        utilised_value = self._coerce_positive_float(target.get("utilised"))
        if isinstance(target.get("utilised"), Mapping):
            utilised_mapping = cast(Mapping[str, Any], target.get("utilised"))
            for key in ("debits", "exposure", "span"):
                candidate = self._coerce_positive_float(utilised_mapping.get(key))
                if candidate is not None:
                    utilised_value = candidate
                    break

        net_value = self._coerce_positive_float(target.get("net"))
        if (
            net_value is None
            and available_value is not None
            and utilised_value is not None
        ):
            net_value = max(available_value + utilised_value, 0.0)

        summary = {
            "available": available_value or 0.0,
            "used": utilised_value or 0.0,
            "net": net_value or (available_value or 0.0),
        }
        return summary

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        """Return ``float`` representation when value is positive.

        Args:
            value: Arbitrary object potentially representing a numeric value.

        Returns:
            float | None: Positive float when coercion succeeds.

        Raises:
            None.
        """

        if isinstance(value, (int, float)):
            result = float(value)
        else:
            result = None
            with suppress(Exception):
                result = float(value)
        if result is None:
            return None
        if not (result > 0):
            return None
        return result

    def get_profile(self) -> dict:
        """Get user profile."""

        self._acquire_bucket(self._GENERAL_BUCKET)
        response = self._ensure_json(self._make_request("GET", "/user/profile"))
        return cast(dict[str, Any], response.get("data", {}))

    def _fetch_instrument_csv(self, exchange: str) -> str:
        """Fetch raw instrument CSV payload for *exchange*.

        Args:
            exchange: Target exchange identifier.

        Returns:
            str: Raw CSV payload for the requested exchange.

        Raises:
            BrokerError: If the CSV content cannot be retrieved.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient._fetch_instrument_csv",
            extra={
                "event": "zerodha.fetch_instrument_csv.enter",
                "exchange": exchange,
            },
        )
        override_keys = [
            f"INSTRUMENTS_URL_{exchange}",
            f"INSTRUMENTS__URL__{exchange}",
        ]
        override_source = ""
        for key in override_keys:
            raw_value = os.getenv(key)
            if raw_value:
                override_source = raw_value.strip()
                break
        if override_source:
            if override_source.lower().startswith(("http://", "https://")):
                try:
                    override_response = httpx.get(
                        override_source, timeout=self._timeout
                    )
                    override_response.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in ZerodhaKiteClient._fetch_instrument_csv: %s",
                        exc,
                        extra={
                            "event": "zerodha.fetch_instrument_csv.http_error",
                            "exchange": exchange,
                            "source": override_source,
                        },
                        exc_info=exc,
                    )
                    raise BrokerError("Instrument download failed") from exc
                LOGGER.info(
                    "Condition met: zerodha.fetch_instrument_csv.http_override",
                    extra={
                        "event": "zerodha.fetch_instrument_csv.http_override",
                        "exchange": exchange,
                        "source": override_source,
                    },
                )
                return override_response.text
            candidate_path = Path(override_source)
            try:
                content = candidate_path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in ZerodhaKiteClient._fetch_instrument_csv: %s",
                    exc,
                    extra={
                        "event": "zerodha.fetch_instrument_csv.path_error",
                        "exchange": exchange,
                        "source": override_source,
                    },
                    exc_info=exc,
                )
                raise BrokerError("Instrument download failed") from exc
            LOGGER.info(
                "Condition met: zerodha.fetch_instrument_csv.path_override",
                extra={
                    "event": "zerodha.fetch_instrument_csv.path_override",
                    "exchange": exchange,
                    "source": override_source,
                },
            )
            return content

        self._acquire_bucket(self._HISTORICAL_BUCKET)
        raw_response = self._make_request(
            "GET",
            f"/instruments/{exchange}",
            raw_response=True,
        )
        if not isinstance(raw_response, httpx.Response):
            LOGGER.error(
                "Failure in ZerodhaKiteClient._fetch_instrument_csv: invalid response",
                extra={
                    "event": "zerodha.fetch_instrument_csv.invalid_response",
                    "exchange": exchange,
                },
            )
            raise BrokerError("Unexpected instrument response type")
        LOGGER.info(
            "Condition met: zerodha.fetch_instrument_csv.default",
            extra={
                "event": "zerodha.fetch_instrument_csv.default",
                "exchange": exchange,
                "status_code": raw_response.status_code,
            },
        )
        return raw_response.text

    def load_instruments(self, exchange: str = _DEFAULT_EXCHANGE) -> list[dict]:
        """Load instrument list for the provided exchange.

        Stores multiple normalized keys per instrument to make lookups resilient.
        """
        LOGGER.debug(
            "Entered ZerodhaKiteClient.load_instruments",
            extra={"event": "zerodha.load_instruments.enter", "exchange": exchange},
        )
        normalized_exchange = (exchange or self._default_exchange).upper()
        try:
            content = self._fetch_instrument_csv(normalized_exchange)
            reader = csv.DictReader(io.StringIO(content))
            instruments = list(reader)
        except BrokerError:
            raise
        except Exception as exc:
            LOGGER.error(
                "Failure in ZerodhaKiteClient.load_instruments: %s",
                exc,
                extra={
                    "event": "zerodha.load_instruments.parse_error",
                    "exchange": normalized_exchange,
                },
                exc_info=exc,
            )
            raise BrokerError("Instrument parse failed") from exc

        # build cache with robust keys:
        cache: dict[str, dict] = {}
        for row in instruments:
            ts_raw = row.get("tradingsymbol") or row.get("symbol") or ""
            if not ts_raw:
                continue
            ts = ts_raw.strip()
            # canonical forms:
            ts_upper = ts.upper()
            ts_nospace = ts_upper.replace(" ", "")
            # exchange-qualified keys (Zerodha style)
            exch = (row.get("exchange") or normalized_exchange or "").strip().upper()
            if exch:
                key_exch = f"{exch}:{ts_upper}"
                cache[key_exch] = row
            # raw trading symbol key (preserve original casing in payload)
            cache[ts_upper] = row
            # nospace fallback (useful when smart_symbol generates no-space versions)
            cache[ts_nospace] = row
            # also register bare base symbol (without exchange prefix) for convenience
            base = ts_upper.split(":", 1)[-1]
            cache[base] = row

        self._instrument_cache[normalized_exchange] = cache

        # Rebuild token_map from all loaded exchanges so callers can do
        # fast O(1) instrument_token -> row lookups without iterating lists.
        for row in instruments:
            try:
                tok = int(row.get("instrument_token") or 0)
            except (TypeError, ValueError):
                continue
            if tok:
                self.token_map[tok] = row

        # [CORRECTED] Aligned correctly
        now = time.time()
        if now - self._last_log_instrument_load >= self._log_throttle_interval:
            LOGGER.info(
                "Condition met: zerodha.load_instruments.success",
                extra={
                    "event": "zerodha.load_instruments.success",
                    "exchange": normalized_exchange,
                    "count": len(cache),
                },
            )
            self._last_log_instrument_load = now
        return instruments

    def instruments(self, exchange: str = "NSE") -> list[dict]:
        """KiteConnect-compatible alias for load_instruments().

        InstrumentManager, get_atm_contracts(), and InstrumentsCache all call
        ``kite.instruments(exchange)`` — this method satisfies that contract.

        Args:
            exchange: Exchange code, e.g. ``"NFO"`` or ``"NSE"``.

        Returns:
            list[dict]: Instrument rows for the requested exchange.

        Raises:
            BrokerError: When the HTTP request or CSV parse fails.
        """
        LOGGER.debug(
            "ZerodhaKiteClient.instruments: delegating to load_instruments exchange=%s",
            exchange,
            extra={"event": "zerodha.instruments.enter", "exchange": exchange},
        )
        return self.load_instruments(exchange)

    def list_instruments(self) -> list[dict[str, Any]]:
        """Return cached instrument rows across all exchanges.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Cached instrument metadata rows.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient.list_instruments",
            extra={"event": "zerodha.list_instruments.enter"},
        )
        try:
            items: list[dict[str, Any]] = []
            for cache in self._instrument_cache.values():
                for row in cache.values():
                    items.append(dict(row))
            LOGGER.info(
                "Condition met: zerodha.list_instruments.success",
                extra={
                    "event": "zerodha.list_instruments.success",
                    "count": len(items),
                },
            )
            return items
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ZerodhaKiteClient.list_instruments: %s",
                exc,
                extra={"event": "zerodha.list_instruments.error"},
                exc_info=exc,
            )
            return []

    def preload_instruments(self) -> None:
        """Preload configured instrument segments into cache.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient.preload_instruments",
            extra={"event": "zerodha.preload_instruments.enter"},
        )
        try:
            bootload_raw = os.getenv("INSTRUMENTS_BOOTLOAD") or os.getenv(
                "INSTRUMENTS__BOOTLOAD"
            )
            bootload_flag = (bootload_raw or "false").strip().lower()
            bootload_enabled = bootload_flag in {"1", "true", "yes", "on"}
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ZerodhaKiteClient.preload_instruments: %s",
                exc,
                extra={"event": "zerodha.preload_instruments.flag_error"},
                exc_info=exc,
            )
            bootload_enabled = False
        if not bootload_enabled:
            LOGGER.info(
                "Condition met: zerodha.preload_instruments.disabled",
                extra={"event": "zerodha.preload_instruments.disabled"},
            )
            return

        segments_env = os.getenv("INSTRUMENTS_ENABLED_SEGMENTS") or os.getenv(
            "INSTRUMENTS__ENABLED_SEGMENTS"
        )
        segments = [
            token.strip().upper()
            for token in (segments_env or "").split(",")
            if token.strip()
        ]
        if not segments:
            segments = [self._default_exchange]

        for segment in segments:
            try:
                self.load_instruments(segment)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in ZerodhaKiteClient.preload_instruments: %s",
                    exc,
                    extra={
                        "event": "zerodha.preload_instruments.load_error",
                        "exchange": segment,
                    },
                    exc_info=exc,
                )
                continue
        LOGGER.info(
            "Condition met: zerodha.preload_instruments.complete",
            extra={
                "event": "zerodha.preload_instruments.complete",
                "segments": segments,
            },
        )

    def get_instrument_token(
        self, symbol: str, exchange: str = _DEFAULT_EXCHANGE
    ) -> int:
        """Get instrument token for symbol with robust fallback attempts."""

        if not symbol:
            raise BrokerError("Missing symbol")

        exchange = (exchange or self._default_exchange).upper()
        raw = str(symbol).strip()
        # potential candidate keys to try (ordered)
        candidates: list[str] = []

        # If symbol already contains an explicit prefix, preserve it as the first candidate
        if ":" in raw:
            cand = raw.upper()
            candidates.append(cand)
            # also push the base-only form
            candidates.append(cand.split(":", 1)[-1])
        else:
            # try exchange-prefixed form
            candidates.append(f"{exchange}:{raw.upper()}")
            candidates.append(raw.upper())

        # space-stripped variant (common mismatch)
        candidates.append(raw.upper().replace(" ", ""))
        # base-only (no exchange)
        candidates.append(raw.upper().split(":", 1)[-1])

        # Ensure unique while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c and c not in seen:
                ordered.append(c)
                seen.add(c)

        if raw.upper() in {"NSE:NIFTY", "NIFTY"}:
            for alias in ("NSE:NIFTY 50", "NIFTY 50", "NIFTY50"):
                if alias not in seen:
                    ordered.append(alias)
                    seen.add(alias)

        # Ensure instrument cache for exchange is loaded
        if exchange not in self._instrument_cache:
            try:
                self.load_instruments(exchange)
            except Exception:
                # load failure will be handled below by fallback to resolver
                pass

        exchange_cache = self._instrument_cache.get(exchange, {})

        for cand in ordered:
            inst = exchange_cache.get(cand)
            if not inst:
                # sometimes stored keys could be without exchange or with different exchange
                # try scanning all caches for an exact match
                for cached_exch, cache in self._instrument_cache.items():
                    if cand in cache:
                        inst = cache[cand]
                        break
            if inst:
                try:
                    return int(inst.get("instrument_token"))
                except Exception:
                    # malformed token; continue to other candidates
                    continue

        # Final fallback: if a resolver is attached, ask it to resolve token
        if self._resolver is not None:
            try:
                token = self._resolver.resolve_symbol_to_token(symbol)
                if token:
                    return int(token)
            except Exception:
                LOGGER.debug(
                    "resolver.resolve_symbol_to_token failed for %s",
                    symbol,
                    exc_info=True,
                )

        raise BrokerError(f"Instrument token not found for {symbol}")

    def close(self) -> None:
        """Close underlying HTTP session."""

        self._client.close()

    def _tokens_to_symbols(
        self, tokens: Iterable[int | str]
    ) -> tuple[list[str], dict[str, int]]:
        """Map instrument tokens OR symbol strings to ``EXCHANGE:SYMBOL`` identifiers.

        ✅ PRODUCTION FIX: Now handles both integer tokens AND symbol strings.
        This supports the PollingStreamer which resolves tokens to symbols before calling.

        Args:
            tokens: Sequence of integer tokens OR symbol strings (e.g., "NSE:NIFTY")

        Returns:
            Tuple of (symbols list, symbol->token mapping)
        """
        resolver = self._resolver
        symbols: list[str] = []
        symbol_map: dict[str, int] = {}

        for token in tokens:
            # ✅ CRITICAL FIX: Handle symbol strings directly
            if isinstance(token, str):
                token_str = token.strip()

                # Check if it's already a valid symbol (contains ":")
                if ":" in token_str:
                    canonical_symbol = token_str.upper()
                    symbols.append(canonical_symbol)
                    # Reverse token enrichment is best-effort only.
                    if resolver is not None:
                        try:
                            if hasattr(resolver, "get_token_for_symbol"):
                                resolved_token = resolver.get_token_for_symbol(
                                    canonical_symbol
                                )
                                if resolved_token:
                                    symbol_map[canonical_symbol] = int(resolved_token)
                            elif hasattr(resolver, "lookup_by_symbol"):
                                info = resolver.lookup_by_symbol(canonical_symbol)
                                if info and "instrument_token" in info:
                                    symbol_map[canonical_symbol] = int(
                                        info["instrument_token"]
                                    )
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.debug(
                                "Reverse token enrichment failed for canonical symbol %s: %s",
                                canonical_symbol,
                                exc,
                                extra={
                                    "event": "reverse_token_enrichment_failed",
                                    "symbol": canonical_symbol,
                                },
                            )
                    continue

                # Check if it's a numeric string (token as string)
                if token_str.isdigit():
                    if resolver is None:
                        symbols.append(token_str)
                        symbol_map[token_str] = int(token_str)
                        continue
                    try:
                        token_int = int(token_str)
                        formatted = resolver.format_token_as_symbol(token_int)
                        if formatted:
                            if ":" not in formatted:
                                formatted = f"{self._default_exchange}:{formatted}"
                            symbols.append(formatted)
                            symbol_map[formatted] = token_int
                        else:
                            symbols.append(token_str)
                            symbol_map[token_str] = token_int
                        continue
                    except (ValueError, TypeError):
                        pass

                # Non-numeric string without ":" - prefix with default exchange
                formatted = f"{self._default_exchange}:{token_str}"
                symbols.append(formatted)
                continue

            # Handle integer tokens (original behavior)
            if resolver is None:
                try:
                    s = str(int(token))
                    symbols.append(s)
                    symbol_map[s] = int(token)
                except (ValueError, TypeError):
                    continue
                continue
            try:
                token_int = int(token)
                formatted = resolver.format_token_as_symbol(token_int)
            except (ValueError, TypeError):
                continue
            except Exception:
                formatted = ""

            if not formatted:
                # last resort: use numeric token string
                try:
                    s = str(int(token))
                    symbols.append(s)
                    symbol_map[s] = int(token)
                except (ValueError, TypeError):
                    continue
                continue

            # ensure canonical form contains exchange prefix
            if ":" not in formatted:
                formatted = f"{self._default_exchange}:{formatted}"
            symbols.append(formatted)
            try:
                symbol_map[formatted] = int(token)
            except (ValueError, TypeError):
                pass

        return symbols, symbol_map

    def _configure_rate_limits(self) -> None:
        """Configure default rate limit buckets."""

        self._limiter.configure_bucket(
            self._QUOTE_BUCKET, capacity=6, refill_rate_per_sec=1.0
        )
        self._limiter.configure_bucket(
            self._ORDER_BUCKET, capacity=10, refill_rate_per_sec=10.0
        )
        self._limiter.configure_bucket(
            self._HISTORICAL_BUCKET, capacity=1, refill_rate_per_sec=1.0
        )
        self._limiter.configure_bucket(
            self._GENERAL_BUCKET, capacity=5, refill_rate_per_sec=5.0
        )

    def _acquire_bucket(self, bucket: str) -> None:
        try:
            # Quotes need a longer timeout to avoid false starvation
            if bucket == self._QUOTE_BUCKET:
                self._limiter.acquire(bucket, timeout=5.0)
            else:
                self._limiter.acquire(bucket, timeout=2.0)

        except RateLimitError as exc:
            snapshot = self._limiter.snapshot()
            raise BrokerError(
                f"Rate limit exceeded for bucket={bucket} | snapshot={snapshot}"
            ) from exc

    def _format_symbol(self, symbol: str) -> str:
        if ":" in symbol:
            return symbol
        return f"{self._default_exchange}:{symbol}"

    def _build_retry_handlers(self, *, endpoint: str) -> tuple[
        Callable[[Exception], bool],
        Callable[[int, Exception, float], None],
    ]:
        """Construct retry decision and callback for broker operations.

        Args:
            endpoint: Endpoint string used for diagnostics and logging.

        Returns:
            tuple[Callable[[Exception], bool], Callable[[int, Exception, float], None]]:
                Retry predicate and retry callback.

        Raises:
            None.
        """

        def _should_retry(exc: Exception) -> bool:
            return isinstance(exc, (RetryableError, httpx.RequestError, RateLimitError))

        def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
            status: int | None = None
            error: Exception | None = None
            endpoint_hint = endpoint
            if isinstance(exc, RetryableError):
                status = exc.context.status
                endpoint_hint = exc.context.endpoint or endpoint_hint
                error = exc.context.error
            elif isinstance(exc, httpx.RequestError):
                error = exc
            self._register_transient_failure(
                status=status,
                error=error,
                delay=delay,
                endpoint=endpoint_hint,
            )
            rotate = False
            if status in {502, 503}:
                rotate = True
            if isinstance(error, httpx.RequestError):
                rotate = True
            if rotate:
                self._rotate_base_url(reason="http_error", status=status)

        return _should_retry, _on_retry

    def _execute_with_retry(
        self,
        *,
        label: str,
        operation: Callable[[], T],
        should_retry: Callable[[Exception], bool],
        error_message: str,
        on_retry: Callable[[int, Exception, float], None] | None,
    ) -> T:
        """Execute broker operation with configured retry policy.

        Args:
            label: Structured label for retry diagnostics.
            operation: Callable representing the broker operation.
            should_retry: Predicate determining retryable exceptions.
            error_message: Message propagated on terminal failure.
            on_retry: Optional callback invoked on each retry attempt.

        Returns:
            T: Result returned by *operation* on success.

        Raises:
            BrokerError: If retries are exhausted without success.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient._execute_with_retry",
            extra={"event": "zerodha_execute_with_retry_start", "label": label},
        )
        try:
            return retry_with_backoff(
                operation=operation,
                retries=min(5, self._max_retries + self._transient_retry_bonus),
                base_delay=self._retry_base_delay,
                max_delay=self._retry_max_delay,
                jitter=self._retry_jitter,
                logger=LOGGER,
                label=label,
                sleep=self._sleep,
                should_retry=should_retry,
                on_retry=on_retry,
            )
        except RetryableError as exc:
            LOGGER.error(
                "Failure in ZerodhaKiteClient._execute_with_retry: %s",
                exc,
                extra={
                    "event": "zerodha_execute_with_retry_start",
                    "label": label,
                    "note": "rate_limit_must_be_acquired_outside",
                },
            )
            raise BrokerError(error_message) from (exc.context.error or exc)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
        *,
        raw_response: bool = False,
        expect_order_response: bool = False,
        operation_label: str | None = None,
        with_retry: bool = True,
    ) -> dict | httpx.Response:
        """Internal method for making API requests with retry handling.

        Args:
            method: HTTP method name such as ``GET`` or ``POST``.
            endpoint: Relative endpoint to invoke on the REST client.
            params: Optional query parameters for the request.
            data: Optional form payload for the request.
            raw_response: Whether to return the raw :class:`httpx.Response`.
            expect_order_response: Flag to tailor error mapping for order APIs.
            operation_label: Structured label used for retry diagnostics.
            with_retry: Toggle to control whether retry logic is applied.

        Returns:
            dict | httpx.Response: Parsed JSON payload or raw response instance.

        Raises:
            BrokerError: If the request exhausts retries or encounters a fatal error.
        """

        self._raise_if_authentication_latched()
        url = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        label = operation_label or (url.lstrip("/") or "zerodha")
        should_retry, on_retry = self._build_retry_handlers(endpoint=url)

        def _operation() -> dict | httpx.Response:
            self._breaker_sleep(url)
            try:
                response = self._client.request(method, url, params=params, data=data)
            except httpx.RequestError as exc:
                raise RetryableError(
                    f"Request error for {url}",
                    context=RetryErrorContext(
                        status=None,
                        endpoint=url,
                        error=exc,
                    ),
                ) from exc

            if raw_response:
                if response.is_success:
                    self._reset_transient_state()
                    return response
                self._reset_transient_state()
                self._raise_for_status(response, expect_order_response)

            if response.is_success:
                # ✅ FIX: Check for empty response body (also transient)
                if not response.content or len(response.content) == 0:
                    raise RetryableError(
                        "Empty response body from Zerodha",
                        context=RetryErrorContext(
                            status=response.status_code,
                            endpoint=url,
                            error=None,
                            delay_hint=0.3,
                        ),
                    )

                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    # ✅ FIX: Make this RETRYABLE instead of terminal
                    raise RetryableError(
                        "Invalid JSON response from Zerodha",
                        context=RetryErrorContext(
                            status=response.status_code,
                            endpoint=url,
                            error=exc,
                            delay_hint=0.5,
                        ),
                    ) from exc
                if self._is_authentication_failure(
                    status_code=response.status_code,
                    payload=payload if isinstance(payload, Mapping) else None,
                    error_text="",
                ):
                    self._mark_authentication_invalid(
                        str(payload.get("message") or payload.get("error_type") or "authentication_failed")
                        if isinstance(payload, Mapping)
                        else "authentication_failed"
                    )
                self._reset_transient_state()
                return payload

            if response.status_code == 429:
                retry_after_raw = response.headers.get("Retry-After")
                delay_hint: float | None = None
                if retry_after_raw is not None:
                    try:
                        delay_hint = float(retry_after_raw)
                    except (TypeError, ValueError):
                        delay_hint = None
                raise RetryableError(
                    f"HTTP 429 for {url}",
                    context=RetryErrorContext(
                        status=response.status_code,
                        endpoint=url,
                        delay_hint=delay_hint,
                    ),
                )

            if response.status_code in {500, 502, 503, 504}:
                raise RetryableError(
                    f"HTTP {response.status_code} for {url}",
                    context=RetryErrorContext(
                        status=response.status_code,
                        endpoint=url,
                    ),
                )

            self._reset_transient_state()
            self._raise_for_status(response, expect_order_response)

        if not with_retry:
            return _operation()

        return self._execute_with_retry(
            label=label,
            operation=_operation,
            should_retry=should_retry,
            error_message="Zerodha request failed",
            on_retry=on_retry,
        )

    def _ensure_json(self, payload: dict[str, Any] | httpx.Response) -> dict[str, Any]:
        """Ensure `_make_request` returned JSON data."""

        if isinstance(payload, httpx.Response):
            raise BrokerError("Unexpected raw HTTP response")
        return payload

    def _raise_for_status(
        self, response: httpx.Response, expect_order_response: bool
    ) -> NoReturn:
        """Convert HTTP response to typed error and raise."""

        message = self._safe_error_message(response)
        status = response.status_code
        payload: Mapping[str, Any] | None = None
        with suppress(Exception):
            raw_payload = response.json()
            if isinstance(raw_payload, Mapping):
                payload = raw_payload
        if self._is_authentication_failure(
            status_code=status,
            payload=payload,
            error_text=message,
        ):
            self._mark_authentication_invalid(message)
        if status in {400, 404} and expect_order_response:
            error: Exception = OrderPlacementError(message)
        elif status == 429:
            error = BrokerError("Zerodha rate limit exceeded")
        else:
            error = BrokerError(message)
        LOGGER.error("Zerodha API error (%s): %s", status, message)
        raise error

    def _safe_error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return f"HTTP {response.status_code}"
        return (
            payload.get("message")
            or payload.get("error_type")
            or f"HTTP {response.status_code}"
        )

    def _sleep(self, delay: float) -> None:
        if delay <= 0:
            return
        time.sleep(delay)

    def _register_transient_failure(
        self,
        *,
        status: int | None,
        error: Exception | None,
        delay: float,
        endpoint: str,
    ) -> None:
        should_open = False
        with self._resilience_lock:
            self._transient_error_streak += 1
            streak = self._transient_error_streak
            if streak >= self._breaker_threshold:
                self._breaker_open_until = max(
                    self._breaker_open_until,
                    time.monotonic() + self._breaker_cooldown_sec,
                )
                self._transient_error_streak = 0
                should_open = True
        status_text = str(status) if status is not None else type(error).__name__
        message = (
            "zerodha_stream_transient status=%s retry_in=%0.2fs consecutive=%d"
            % (status_text, delay, streak)
        )
        self._log_transient(message, level=logging.DEBUG, endpoint=endpoint)
        if should_open:
            self._log_transient(
                "zerodha_stream_circuit_open sleep=%0.1fs threshold=%d"
                % (self._breaker_cooldown_sec, self._breaker_threshold),
                level=logging.WARNING,
                force=True,
            )

    def _log_transient(
        self,
        message: str,
        *,
        level: int = logging.WARNING,
        force: bool = False,
        endpoint: str | None = None,
    ) -> None:
        now = self._log_time_fn()
        should_log = True
        if self._log_cooldown_sec > 0 and not force:
            with self._resilience_lock:
                if now - self._last_transient_log < self._log_cooldown_sec:
                    should_log = False
                else:
                    self._last_transient_log = now
        elif force:
            with self._resilience_lock:
                self._last_transient_log = now
        if should_log:
            extra = {"endpoint": endpoint} if endpoint else None
            LOGGER.log(level, message, extra=extra)

    def _breaker_sleep(self, endpoint: str) -> None:
        with self._resilience_lock:
            open_until = self._breaker_open_until
        if open_until <= 0.0:
            return
        remaining = max(0.0, open_until - time.monotonic())
        if remaining <= 0.0:
            with self._resilience_lock:
                self._breaker_open_until = 0.0
            return
        self._log_transient(
            "zerodha_stream_circuit_open_skip remaining=%0.1fs endpoint=%s"
            % (remaining, endpoint),
            level=logging.WARNING,
            force=False,
            endpoint=endpoint,
        )
        raise RetryableError(
            f"Circuit open for {remaining:.1f}s, skipping {endpoint}",
            context=RetryErrorContext(
                status=None,
                endpoint=endpoint,
                delay_hint=remaining,
            ),
        )

    def _reset_transient_state(self) -> None:
        with self._resilience_lock:
            self._transient_error_streak = 0
            self._breaker_open_until = 0.0
        if self._auth_invalid:
            # An authenticated request just succeeded: the console/token was
            # fixed. Clear the latch so trading re-arms without a restart.
            self._auth_invalid = False
            self._auth_invalid_reason = None
            self._auth_invalid_at = None
            self._auth_reprobe_next = 0.0
            LOGGER.warning(
                "ZERODHA_AUTH_RESTORED generation=%s",
                self._auth_failure_generation,
                extra={
                    "event": "ZERODHA_AUTH_RESTORED",
                    "generation": self._auth_failure_generation,
                },
            )

    def _create_http_client(self, base_url: str) -> httpx.Client:
        # Force outbound connections over IPv4. Zerodha's developer console
        # allowlists the static IPv4 (15.206.3.6), but the host can otherwise
        # reach Zerodha over IPv6, which is NOT allowlisted -> 403 "IP is not
        # allowed to place orders". Binding the local address to the IPv4 stack
        # makes every request present the allowlisted IPv4. Opt out with
        # ZERODHA_FORCE_IPV4=false if ever needed.
        transport = None
        force_ipv4 = str(os.getenv("ZERODHA_FORCE_IPV4", "true")).strip().lower() in {"1", "true", "yes", "on"}
        if force_ipv4:
            try:
                transport = httpx.HTTPTransport(local_address="0.0.0.0")
                LOGGER.info(
                    "ZERODHA_HTTP_IPV4_FORCED enabled=True local_address=0.0.0.0",
                    extra={"event": "ZERODHA_HTTP_IPV4_FORCED", "enabled": True, "local_address": "0.0.0.0"},
                )
            except Exception:  # pragma: no cover - defensive
                transport = None
        else:
            LOGGER.info(
                "ZERODHA_HTTP_IPV4_FORCED enabled=False local_address=None",
                extra={"event": "ZERODHA_HTTP_IPV4_FORCED", "enabled": False, "local_address": None},
            )
        return httpx.Client(
            base_url=base_url,
            timeout=self._timeout,
            transport=transport,
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {self._api_key}:{self._access_token}",
            },
        )

    def _update_http_client(self, base_url: str) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("zerodha_client_close_failed", exc_info=True)
        self._base_url = base_url.rstrip("/")
        self._client = self._create_http_client(self._base_url)

    def _rotate_base_url(self, *, reason: str | None, status: int | None) -> None:
        if len(self._base_urls) <= 1:
            return
        with self._resilience_lock:
            self._base_index = (self._base_index + 1) % len(self._base_urls)
            next_base = self._base_urls[self._base_index]
        LOGGER.warning(
            "zerodha_base_url_rotate",
            extra={
                "event": "zerodha_base_url_rotate",
                "reason": reason,
                "status": status,
                "base_url": next_base,
            },
        )
        self._update_http_client(next_base)


class ZerodhaKiteWebSocket:
    """WebSocket client for live market data with resilient queuing."""

    def __init__(
        self,
        api_key: str,
        access_token: str,
        on_tick: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        if not api_key or not access_token:
            raise ConfigurationError("WebSocket credentials missing")

        # Force websocket-client to prefer IPv4. Some container networks hang on AAAA.
        os.environ["WEBSOCKET_CLIENT_ENABLE_IPV6"] = "0"

        self._api_key = api_key.strip()
        self._access_token = _sanitize_access_token(access_token)
        if not self._access_token:
            raise ConfigurationError("WebSocket access token missing")

        self._ticker: KiteTickerType | None = None
        self._lock = threading.RLock()
        self._connected = False
        self._connecting = False
        self._last_error: WebSocketError | None = None

        self._on_tick_cb = on_tick or (lambda _tick: None)
        self._on_error_cb = on_error or (lambda _exc: None)
        self._on_close_cb: Optional[Callable[..., None]] = None
        self._on_reconnect_cb: Optional[Callable[..., None]] = None
        self._on_open_cb: Optional[Callable[..., None]] = None

        self._pending_subs: set[int] = set()
        self._pending_unsubs: set[int] = set()
        self._pending_modes: dict[str, set[int]] = {}

        self._connect_timeout = self._parse_positive_int(
            os.getenv("ZERODHA_WS_CONNECT_TIMEOUT"), fallback=45, minimum=10
        )
        self._reconnect_max_delay = self._parse_positive_int(
            os.getenv("ZERODHA_WS_RECONNECT_MAX_DELAY"),
            fallback=getattr(KiteTicker, "RECONNECT_MAX_DELAY", 60),
            minimum=5,
        )
        self._reconnect_max_tries = self._parse_positive_int(
            os.getenv("ZERODHA_WS_RECONNECT_MAX_TRIES"),
            fallback=getattr(KiteTicker, "RECONNECT_MAX_TRIES", 50),
            minimum=5,
        )
        self._disable_ssl = self._parse_bool(os.getenv("ZERODHA_WS_DISABLE_SSL"))
        self._proxy = self._build_proxy()
        origin = (os.getenv("ZERODHA_WS_ORIGIN") or "https://kite.zerodha.com").strip()
        if not origin:
            origin = "https://kite.zerodha.com"
        self._connect_headers: dict[str, str] = {"Origin": origin}

    # ---- public API consumed by WebSocketManager
    def connect(self, threaded: bool = True, **kwargs: Any) -> None:
        """Connect to the websocket, queueing operations until ready."""

        if KiteTicker is None:  # pragma: no cover - runtime guard
            raise ConfigurationError("kiteconnect is not installed")

        with self._lock:
            if self._connected:
                LOGGER.debug("Zerodha WS connect skipped: already connected")
                return
            if self._connecting:
                LOGGER.debug("Zerodha WS connect skipped: already connecting")
                return
            self._connecting = True
            self._last_error = None
            self._ticker = self._build_ticker()
            self._wire_callbacks(self._ticker)

        ticker = self._ticker
        if ticker is None:  # pragma: no cover - defensive guard
            with self._lock:
                self._connecting = False
            raise WebSocketError("Ticker client unavailable")

        connect_kwargs: dict[str, Any] = dict(kwargs)
        connect_kwargs.setdefault("threaded", threaded)
        connect_kwargs.setdefault("ping_interval", 20)
        connect_kwargs.setdefault("ping_timeout", 10)

        if self._disable_ssl:
            connect_kwargs.setdefault("disable_ssl_verification", True)
        if self._proxy is not None:
            connect_kwargs.setdefault("proxy", self._proxy)

        headers = self._merge_headers(connect_kwargs.pop("header", None))
        if headers:
            connect_kwargs["header"] = headers

        header_keys = sorted(
            {
                header.split(":", 1)[0].strip().lower()
                for header in headers
                if ":" in header
            }
        )
        LOGGER.debug(
            "Zerodha WS headers merged",
            extra={"count": len(headers), "keys": header_keys},
        )

        LOGGER.info(
            "Connecting Zerodha websocket (threaded=%s, ping=%ss/%ss, headers=%d)",
            connect_kwargs.get("threaded"),
            connect_kwargs.get("ping_interval"),
            connect_kwargs.get("ping_timeout"),
            len(headers),
        )

        try:
            try:
                ticker.connect(**connect_kwargs)
            except TypeError:
                legacy_kwargs: dict[str, Any] = {
                    "threaded": connect_kwargs.get("threaded", threaded)
                }
                LOGGER.debug(
                    "Retrying KiteTicker.connect with legacy kwargs",
                    extra={"kwargs": legacy_kwargs},
                )
                ticker.connect(**legacy_kwargs)
        except Exception as exc:  # noqa: BLE001
            error = WebSocketError(f"Failed to connect Zerodha websocket: {exc}")
            setattr(error, "code", getattr(exc, "code", None))
            setattr(error, "reason", getattr(exc, "reason", None))
            self._record_error(error)
            with self._lock:
                self._connecting = False
                self._connected = False
            raise error from exc

    def disconnect(self) -> None:
        """Disconnect the websocket client and clear state."""

        with self._lock:
            ticker = self._ticker
            self._ticker = None
            self._connected = False
            self._connecting = False

        if ticker is None:
            return

        try:
            ticker.close()
        except Exception:  # pragma: no cover - best effort cleanup
            LOGGER.debug("Exception during websocket close", exc_info=True)

    def close(self) -> None:
        """Alias for :meth:`disconnect` for compatibility with KiteTicker."""

        self.disconnect()

    def refresh_session(self) -> None:
        """Refresh underlying Kite session before reconnecting."""

        if KiteTicker is None:  # pragma: no cover - runtime guard
            raise ConfigurationError("kiteconnect is not installed")

        refreshed = _sanitize_access_token(
            os.getenv("ZERODHA_ACCESS_TOKEN") or self._access_token
        )
        old_ticker: KiteTickerType | None = None

        with self._lock:
            if refreshed and refreshed != self._access_token:
                LOGGER.info(
                    "WS session refreshed: using new access token (len=%d)",
                    len(refreshed),
                )
                self._access_token = refreshed
            old_ticker = self._ticker
            self._ticker = self._build_ticker()
            self._wire_callbacks(self._ticker)
            self._connected = False
            self._connecting = False

        if old_ticker is not None:
            try:
                old_ticker.close()
            except Exception:  # pragma: no cover - cleanup best effort
                LOGGER.debug(
                    "Exception closing previous ticker during refresh", exc_info=True
                )

    def subscribe(self, instrument_tokens: Iterable[int]) -> None:
        """Subscribe to instruments, deferring if socket is not ready."""

        tokens = [int(token) for token in instrument_tokens if token is not None]
        if not tokens:
            return

        with self._lock:
            self._pending_unsubs.difference_update(tokens)
            if self._connected and self._ticker is not None:
                try:
                    self._chunked_call(self._ticker.subscribe, tokens)
                    return
                except Exception as exc:
                    LOGGER.warning("WS subscribe failed (direct): %s", exc)
            self._pending_subs.update(tokens)

    def unsubscribe(self, instrument_tokens: Iterable[int]) -> None:
        """Unsubscribe from instruments, deferring if socket is not ready."""

        tokens = [int(token) for token in instrument_tokens if token is not None]
        if not tokens:
            return

        with self._lock:
            self._pending_subs.difference_update(tokens)
            for mode_tokens in self._pending_modes.values():
                mode_tokens.difference_update(tokens)
            if self._connected and self._ticker is not None:
                try:
                    self._chunked_call(self._ticker.unsubscribe, tokens)
                    return
                except Exception as exc:
                    LOGGER.warning("WS unsubscribe failed (direct): %s", exc)
            self._pending_unsubs.update(tokens)

    def set_mode(self, mode: str, instrument_tokens: Iterable[int]) -> None:
        """Set streaming mode (ltp, quote, full)."""

        tokens = {int(token) for token in instrument_tokens if token is not None}
        if not tokens:
            return

        with self._lock:
            resolved_mode = None
            if self._ticker is not None:
                resolved_mode = self._resolve_mode(self._ticker, mode)
            if resolved_mode is None:
                resolved_mode = mode
            if self._connected and self._ticker is not None:
                try:
                    actual_mode = self._resolve_mode(self._ticker, resolved_mode)
                    if actual_mode is None:
                        raise WebSocketError(f"Unsupported websocket mode: {mode}")
                    self._ticker.set_mode(actual_mode, list(tokens))
                    return
                except Exception as exc:
                    LOGGER.warning("WS set_mode failed (direct): %s", exc)
            self._pending_modes.setdefault(resolved_mode, set()).update(tokens)

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def is_connecting(self) -> bool:
        with self._lock:
            return self._connecting

    def pending_counts(self) -> dict[str, int]:
        with self._lock:
            modes_count = sum(len(tokens) for tokens in self._pending_modes.values())
            return {
                "subs": len(self._pending_subs),
                "unsubs": len(self._pending_unsubs),
                "modes": modes_count,
            }

    def last_error(self) -> WebSocketError | None:
        return self._last_error

    # ---- internal
    def _build_ticker(self) -> KiteTickerType:
        return KiteTicker(
            self._api_key,
            self._access_token,
            reconnect=True,
            reconnect_max_tries=self._reconnect_max_tries,
            reconnect_max_delay=self._reconnect_max_delay,
            connect_timeout=self._connect_timeout,
        )

    def _wire_callbacks(self, ticker: KiteTickerType) -> None:
        def _on_connect_adapter(*args: Any, **kwargs: Any) -> None:
            LOGGER.info("Zerodha websocket connected")
            with self._lock:
                self._connected = True
                self._connecting = False
            self._flush_pending()
            if self._on_reconnect_cb is not None:
                try:
                    self._on_reconnect_cb(*args, **kwargs)
                except Exception:  # pragma: no cover - user callback
                    LOGGER.exception("user on_reconnect raised")
            if self._on_open_cb is not None:
                try:
                    self._on_open_cb(*args, **kwargs)
                except Exception:  # pragma: no cover - user callback
                    LOGGER.exception("user on_open raised")

        def _on_ticks_adapter(*args: Any, **kwargs: Any) -> None:
            payload = kwargs.get("ticks")
            if payload is None and args:
                payload = args[-1]
            try:
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            self._on_tick_cb(item)
                elif isinstance(payload, dict):
                    self._on_tick_cb(payload)
            except Exception as exc:  # noqa: BLE001 - guard user callback
                LOGGER.exception("on_ticks adapter error: %s", exc)
                try:
                    self._on_error_cb(exc)
                except Exception:  # pragma: no cover - avoid cascading errors
                    LOGGER.exception("user on_error callback raised")

        def _on_error_adapter(*args: Any, **kwargs: Any) -> None:
            code = kwargs.get("code")
            reason = kwargs.get("reason")
            if code is None or reason is None:
                tail = list(args)[-2:] if len(args) >= 2 else list(args)
                if len(tail) == 2:
                    code, reason = tail
                elif len(tail) == 1:
                    code = tail[0]

            error = WebSocketError(
                f"Zerodha websocket error code={code!r} reason={reason!r}"
            )
            setattr(error, "code", code)
            setattr(error, "reason", reason)

            LOGGER.error("WebSocket error: code=%s reason=%s", code, reason)
            self._record_error(error)
            with self._lock:
                self._connecting = False
            try:
                self._on_error_cb(error)
            except Exception:  # pragma: no cover - user callback
                LOGGER.exception("user on_error raised")

        def _on_close_adapter(*args: Any, **kwargs: Any) -> None:
            code = kwargs.get("code")
            reason = kwargs.get("reason")
            with self._lock:
                self._connected = False
                self._connecting = False
            LOGGER.info("WebSocket closed: code=%s reason=%s", code, reason)
            if self._on_close_cb is not None:
                try:
                    self._on_close_cb(*args, **kwargs)
                except Exception:  # pragma: no cover - user callback
                    LOGGER.exception("user on_close raised")

        ticker.on_connect = _on_connect_adapter
        ticker.on_ticks = _on_ticks_adapter
        ticker.on_error = _on_error_adapter
        ticker.on_close = _on_close_adapter
        if hasattr(ticker, "on_noreconnect"):

            def _on_no_reconnect(*_: Any, **__: Any) -> None:
                error = WebSocketError(
                    "Zerodha websocket exhausted reconnection attempts"
                )
                self._record_error(error)
                LOGGER.error("%s", error)
                try:
                    self._on_error_cb(error)
                except Exception:  # pragma: no cover - user callback
                    LOGGER.exception("user on_error raised")

            ticker.on_noreconnect = _on_no_reconnect

    def _flush_pending(self) -> None:
        with self._lock:
            if not self._connected or self._ticker is None:
                return

            subs = list(self._pending_subs)
            unsubs = list(self._pending_unsubs)
            modes = {mode: list(tokens) for mode, tokens in self._pending_modes.items()}

            self._pending_subs.clear()
            self._pending_unsubs.clear()
            self._pending_modes.clear()

            ticker = self._ticker

        modes_count = sum(len(tokens) for tokens in modes.values())
        LOGGER.info(
            "Flushing pending subscriptions",
            extra={
                "subs": len(subs),
                "unsubs": len(unsubs),
                "modes": modes_count,
            },
        )

        try:
            if subs:
                self._chunked_call(ticker.subscribe, subs)
            for mode, tokens in modes.items():
                if tokens:
                    actual_mode = self._resolve_mode(ticker, mode)
                    if actual_mode is None:
                        LOGGER.warning(
                            "Skipping unsupported websocket mode during flush: %s", mode
                        )
                        continue
                    ticker.set_mode(actual_mode, tokens)
            if unsubs:
                self._chunked_call(ticker.unsubscribe, unsubs)
        except Exception as exc:
            LOGGER.warning("WS flush_pending failed: %s", exc)

    def _chunked_call(
        self, func: Callable[[list[int]], Any], tokens: Iterable[int]
    ) -> None:
        token_list = list(tokens)
        for idx in range(0, len(token_list), 400):
            chunk = token_list[idx : idx + 400]
            func(chunk)
            if idx + 400 < len(token_list):
                time.sleep(0.05)

    def _resolve_mode(self, ticker: KiteTickerType, mode: str) -> str | None:
        mapping = {
            "ltp": getattr(ticker, "MODE_LTP", "ltp"),
            "quote": getattr(ticker, "MODE_QUOTE", "quote"),
            "full": getattr(ticker, "MODE_FULL", "full"),
        }
        if mode in mapping.values():
            return mode
        lowered = mode.lower()
        if lowered.startswith("mode_"):
            lowered = lowered.split("mode_", 1)[1]
        return mapping.get(lowered)

    @staticmethod
    def _parse_positive_int(value: str | None, *, fallback: int, minimum: int) -> int:
        if value is None:
            return max(fallback, minimum)
        try:
            parsed = int(float(value))
        except ValueError:
            return max(fallback, minimum)
        if parsed <= 0:
            return minimum
        return max(parsed, minimum)

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _build_proxy(self) -> dict[str, Any] | None:
        host = os.getenv("ZERODHA_WS_PROXY_HOST")
        port = os.getenv("ZERODHA_WS_PROXY_PORT")
        if not host or not port:
            return None
        try:
            port_int = int(float(port))
        except ValueError:
            LOGGER.warning("Invalid Zerodha WS proxy port: %s", port)
            return None
        if port_int <= 0:
            LOGGER.warning("Proxy port must be positive: %s", port)
            return None
        return {"host": host, "port": port_int}

    def _merge_headers(self, raw: Any) -> list[str]:
        headers: list[str] = []
        header_keys: set[str] = set()

        def _add(key: str, value: str) -> None:
            key_clean = key.strip()
            if not key_clean:
                return
            header_keys.add(key_clean.lower())
            headers.append(f"{key_clean}: {value.strip()}")

        if raw:
            if isinstance(raw, dict):
                for key, value in raw.items():
                    _add(str(key), str(value))
            else:
                items: Iterable[Any]
                if isinstance(raw, (str, bytes)):
                    items = [raw]
                elif isinstance(raw, Iterable):
                    items = raw
                else:
                    items = [raw]
                for entry in items:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        _add(str(entry[0]), str(entry[1]))
                    elif isinstance(entry, str) and ":" in entry:
                        key, value = entry.split(":", 1)
                        _add(key, value)

        for key, value in self._connect_headers.items():
            if key.lower() not in header_keys:
                _add(key, value)

        if "origin" not in header_keys:
            _add(
                "Origin",
                self._connect_headers.get("Origin", "https://kite.zerodha.com"),
            )

        return headers

    def _record_error(self, error: WebSocketError) -> None:
        self._last_error = error

    # Properties used by external callers to attach handlers
    @property
    def on_tick(self) -> Callable[[dict[str, Any]], None]:
        return self._on_tick_cb

    @on_tick.setter
    def on_tick(self, callback: Optional[Callable[[dict[str, Any]], None]]) -> None:
        self._on_tick_cb = callback or (lambda _tick: None)

    @property
    def on_error(self) -> Callable[[Exception], None]:
        return self._on_error_cb

    @on_error.setter
    def on_error(self, callback: Optional[Callable[[Exception], None]]) -> None:
        self._on_error_cb = callback or (lambda _exc: None)

    @property
    def on_close(self) -> Optional[Callable[..., None]]:
        return self._on_close_cb

    @on_close.setter
    def on_close(self, callback: Optional[Callable[..., None]]) -> None:
        self._on_close_cb = callback

    @property
    def on_reconnect(self) -> Optional[Callable[..., None]]:
        return self._on_reconnect_cb

    @on_reconnect.setter
    def on_reconnect(self, callback: Optional[Callable[..., None]]) -> None:
        self._on_reconnect_cb = callback

    @property
    def on_open(self) -> Optional[Callable[..., None]]:
        return self._on_open_cb

    @on_open.setter
    def on_open(self, callback: Optional[Callable[..., None]]) -> None:
        self._on_open_cb = callback


__all__ = ["ZerodhaKiteClient", "ZerodhaKiteWebSocket"]
