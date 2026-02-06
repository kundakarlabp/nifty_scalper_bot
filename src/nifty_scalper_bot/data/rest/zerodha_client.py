"""Production ready Zerodha Kite REST and websocket clients."""

from __future__ import annotations

from contextlib import suppress
import csv
from dataclasses import dataclass
from datetime import datetime
import io
import json
import logging
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

    from nifty_scalper_bot.data.instruments import InstrumentResolver
else:  # pragma: no cover - fallback type when kiteconnect missing
    KiteTickerType = Any
    InstrumentResolver = Any  # type: ignore[misc]

from nifty_scalper_bot.data.rest.client import BaseBrokerClient
from nifty_scalper_bot.utils.env import get_float, get_int, get_str
from nifty_scalper_bot.utils.errors import (
    BrokerError,
    ConfigurationError,
    OrderPlacementError,
    WebSocketError,
)
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import RateLimiter, RateLimitError
from nifty_scalper_bot.utils.retry import (
    RetryableError,
    RetryErrorContext,
    retry_with_backoff,
)

T = TypeVar("T")

LOGGER = get_logger(__name__)


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
        self._last_log_instrument_load = 0.0
        self._rest_cache_ttl = max(
            1.0, get_float("BROKER_REST_CACHE_TTL_SEC", default=15.0)
        )
        self._positions_cache: _RestCacheEntry | None = None
        self._margins_cache: dict[str, _RestCacheEntry] = {}

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
                    candidate = "NSE:NIFTY 50"
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
            LOGGER.error(
                "Failure in ZerodhaKiteClient.quote_any request: %s",
                exc,
                extra={"event": "zerodha_quote_any_request_error"},
                exc_info=exc,
            )
            return None

        data = response.get("data")
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
        """Get quote for symbol."""

        self._acquire_bucket(self._QUOTE_BUCKET)
        kite_symbol = self._format_symbol(symbol)
        response = self._ensure_json(
            self._make_request("GET", "/quote", params={"i": [kite_symbol]})
        )
        try:
            quote_data = response["data"][kite_symbol]
        except KeyError as exc:  # pragma: no cover - API invariant
            raise BrokerError(f"Quote data missing for {symbol}") from exc

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

        # [FIX] Filter out None values
        clean_params = {k: v for k, v in params.items() if v is not None}
        if clean_params.get("order_type") == "MARKET":
            clean_params.pop("trigger_price", None)

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
            return {
                "order_id": data.get("order_id"),
                "status": response.get("status", "success"),
                "message": response.get("message", ""),
                "tag": final_tag,
            }

        except Exception as e:
            # [FIX] Add specific logging for 405 errors
            if "405" in str(e):
                self._logger.critical(
                    f"🛑 Zerodha 405 Error (Bad URL/Method). URL: /orders/{variety}, Method: POST"
                )

            # [FIX] Fail Fast Logic
            from nifty_scalper_bot.utils.errors import OrderPlacementError

            raise OrderPlacementError(f"Order placement failed: {e}")

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
                for token, payload in quote_map.items():
                    if not isinstance(payload, Mapping):
                        continue
                    try:
                        last_price = float(payload.get("last_price", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue

                    # [CORRECTED] Indentation fixed to be inside the for loop
                    if last_price > 0:
                        out[token] = last_price

                # [CORRECTED] Logic for returning data if found (inside try)
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

                # depth fetch returned but no usable last_price values
                LOGGER.warning(
                    "zerodha_get_ltp_bulk_depth_no_ltp",
                    extra={
                        "event": "zerodha_get_ltp_bulk_depth_no_ltp",
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
            tokens: List of integer tokens OR symbol strings (e.g., ["NSE:NIFTY 50"])

        Returns:
            Dict mapping symbol strings to quote payloads
        """
        if not tokens:
            return {}

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

        # ✅ FIX: Return keyed by symbol (not token) for consistency
        out: dict[str, dict[str, Any]] = {}
        for symbol, payload in data.items():
            if not payload:
                continue
            # Ensure instrument_token is present
            if "instrument_token" not in payload:
                token_from_map = symbol_map.get(symbol)
                if token_from_map:
                    payload["instrument_token"] = token_from_map
            out[symbol] = payload

        return out

    def quote(self, instruments: list[str] | list[int] | str | int) -> dict[str, Any]:
        """Standard KiteConnect compliant alias for quote fetching.

        ✅ PRODUCTION FIX: Now accepts both symbol strings AND integer tokens.

        Args:
            instruments: List of symbols (e.g., ["NSE:NIFTY 50"]) OR tokens OR single value

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
        # 1. Format Dates (Handle datetime objects vs strings)
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d %H:%M:%S")

        # 2. Build Parameters
        params = {
            "from": from_date,
            "to": to_date,
            "continuous": 1 if continuous else 0,
            "oi": 1 if oi else 0,
        }

        # 3. Execute Request (Using your native infrastructure)
        self._acquire_bucket(self._HISTORICAL_BUCKET)
        response = self._ensure_json(
            self._make_request(
                "GET",
                f"/instruments/historical/{instrument_token}/{interval}",
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
        for order in orders:
            if order.get("order_id") == order_id:
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
            raise

    def get_positions(self) -> list[dict[str, Any]]:
        """Return Zerodha positions (Log-Silent if empty)."""
        LOGGER.debug(
            "Entered ZerodhaKiteClient.get_positions",
            extra={"event": "zerodha_get_positions_start"},
        )
        label = "positions.fetch"
        endpoint = "/portfolio/positions"
        should_retry, on_retry = self._build_retry_handlers(endpoint=endpoint)

        def _operation() -> list[dict[str, Any]]:
            self._acquire_bucket(self._GENERAL_BUCKET)
            response = self._ensure_json(
                self._make_request(
                    "GET", endpoint, operation_label=label, with_retry=False
                )
            )
            payload = response.get("data")
            normalized: list[dict[str, Any]] = []

            # Case A: Payload is a Dict (net/day keys)
            if isinstance(payload, dict):
                for key in ("net", "day"):
                    positions = payload.get(key)
                    if isinstance(positions, list):
                        normalized = cast(list[dict[str, Any]], positions)
                        if normalized:  # If found valid list, stop looking
                            break

            # Case B: Payload is a List (direct)
            elif isinstance(payload, list):
                normalized = cast(list[dict[str, Any]], payload)

            # [FIX] Consolidated Logging for all cases
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
            cached = self._load_rest_cache(self._positions_cache, label=label)
            if cached is not None:
                return cast(list[dict[str, Any]], cached)
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
        """Return available margin balance for a Zerodha segment.

        Args:
            segment: Zerodha segment identifier such as ``"equity"``.

        Returns:
            float: Available balance extracted from the margin payload.

        Raises:
            BrokerError: Propagated when the margin fetch fails.
        """

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
        summary: dict[str, float] | None = None
        account_error: Exception | None = None
        try:
            account_payload = self.get_account_margins(segment=normalized_segment)
            if account_payload:
                summary = self._normalize_margin_payload(
                    account_payload,
                    segment=normalized_segment,
                )
        except Exception as exc:  # noqa: BLE001 - continue with fallback
            account_error = exc
            LOGGER.error(
                "Failure in ZerodhaKiteClient.get_available_balance account fetch: %s",
                exc,
                extra={
                    "event": "zerodha_available_balance_account_error",
                    "segment": normalized_segment,
                },
                exc_info=exc,
            )

        if not summary:
            try:
                summary = self.get_margin_summary(segment=normalized_segment)
                if account_error is not None:
                    LOGGER.info(
                        "Condition met: zerodha_available_balance_summary_fallback",
                        extra={
                            "event": "zerodha_available_balance_summary_fallback",
                            "segment": normalized_segment,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - fallback to env
                LOGGER.error(
                    "Failure in ZerodhaKiteClient.get_available_balance: %s",
                    exc,
                    extra={
                        "event": "zerodha_available_balance_error",
                        "segment": normalized_segment,
                    },
                    exc_info=exc,
                )
                fallback_balance = self._resolve_balance_fallback()
                LOGGER.warning(
                    "Condition met: zerodha_available_balance_env_fallback",
                    extra={
                        "event": "zerodha_available_balance_env_fallback",
                        "segment": normalized_segment,
                        "fallback": fallback_balance,
                    },
                )
                return fallback_balance

        summary = summary or {"available": 0.0}
        available = float(summary.get("available", 0.0) or 0.0)
        if available <= 0.0:
            alternate_keys = ("available_cash", "cash", "net")
            for key in alternate_keys:
                raw_value = summary.get(key)
                if raw_value is None:
                    continue
                try:
                    candidate = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if candidate > 0.0:
                    available = candidate
                    LOGGER.info(
                        "Condition met: zerodha_available_balance_alternate_key",
                        extra={
                            "event": "zerodha_available_balance_alternate_key",
                            "segment": normalized_segment,
                            "key": key,
                            "available": available,
                        },
                    )
                    break

        if available <= 0.0:
            fallback_balance = self._resolve_balance_fallback()
            LOGGER.warning(
                "Condition met: zerodha_available_balance_non_positive",
                extra={
                    "event": "zerodha_available_balance_non_positive",
                    "segment": normalized_segment,
                    "available": available,
                    "fallback": fallback_balance,
                },
            )
            return fallback_balance

        # [CORRECTED] Aligned correctly and placed before return
        now = time.time()
        if now - self._last_log_balance >= self._log_throttle_interval:
            # [FIX] Keep as INFO, throttled by _log_throttle_interval (default 60s)
            LOGGER.info(
                "zerodha_available_balance_success",
                extra={
                    "event": "zerodha_available_balance_success",
                    "segment": normalized_segment,
                    "available": available,
                },
            )
            self._last_log_balance = now
        return available

    def _resolve_balance_fallback(self) -> float:
        """Return environment configured fallback balance figure.

        Args:
            None.

        Returns:
            float: Non-negative fallback balance from environment variables.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ZerodhaKiteClient._resolve_balance_fallback",
            extra={"event": "zerodha_balance_fallback_enter"},
        )
        fallback_value = 1_000_000.0
        try:
            candidates = (
                os.getenv("RISK__CAPITAL"),
                os.getenv("RISK_CAPITAL"),
                os.getenv("BACKTEST__CAPITAL"),
            )
            for candidate in candidates:
                if candidate is None:
                    continue
                token = candidate.strip()
                if not token:
                    continue
                fallback_value = max(float(token), 0.0)
                break
        except Exception as exc:  # noqa: BLE001 - defensive parsing
            LOGGER.error(
                "Failure in ZerodhaKiteClient._resolve_balance_fallback: %s",
                exc,
                extra={"event": "zerodha_balance_fallback_error"},
                exc_info=exc,
            )
            fallback_value = 1_000_000.0
        LOGGER.info(
            "Condition met: zerodha_balance_fallback_resolved",
            extra={
                "event": "zerodha_balance_fallback_resolved",
                "fallback": fallback_value,
            },
        )
        return fallback_value

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
            tokens: Sequence of integer tokens OR symbol strings (e.g., "NSE:NIFTY 50")

        Returns:
            Tuple of (symbols list, symbol->token mapping)
        """
        resolver = self._resolver
        symbols: list[str] = []
        symbol_map: dict[str, int] = {}

        if resolver is None:
            return symbols, symbol_map

        for token in tokens:
            # ✅ CRITICAL FIX: Handle symbol strings directly
            if isinstance(token, str):
                token_str = token.strip()

                # Check if it's already a valid symbol (contains ":")
                if ":" in token_str:
                    symbols.append(token_str)
                    # Try to get the token from resolver for the reverse map
                    try:
                        if hasattr(resolver, "get_token_for_symbol"):
                            resolved_token = resolver.get_token_for_symbol(token_str)
                            if resolved_token:
                                symbol_map[token_str] = int(resolved_token)
                        elif hasattr(resolver, "lookup_by_symbol"):
                            info = resolver.lookup_by_symbol(token_str)
                            if info and "instrument_token" in info:
                                symbol_map[token_str] = int(info["instrument_token"])
                    except Exception:
                        pass  # Token lookup failed, but we can still use the symbol
                    continue

                # Check if it's a numeric string (token as string)
                if token_str.isdigit():
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
            return isinstance(exc, (RetryableError, httpx.RequestError))

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
                retries=self._max_retries + self._transient_retry_bonus,
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
        error: Exception
        if status in {400, 404} and expect_order_response:
            error = OrderPlacementError(message)

        if status == 400:
            # LOG THE FULL PAYLOAD IF POSSIBLE
            LOGGER.error(f"🛑 Zerodha 400 Bad Request: {message}")
        elif status == 401:
            error = ConfigurationError("Zerodha authentication failed")
        elif status == 403:
            error = ConfigurationError("Zerodha access denied")
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
            return
        self._log_transient(
            "zerodha_stream_circuit_sleep remaining=%0.1fs endpoint=%s"
            % (remaining, endpoint),
            level=logging.WARNING,
            force=True,
            endpoint=endpoint,
        )
        self._sleep(remaining)
        with self._resilience_lock:
            self._breaker_open_until = 0.0

    def _reset_transient_state(self) -> None:
        with self._resilience_lock:
            self._transient_error_streak = 0
            self._breaker_open_until = 0.0

    def _create_http_client(self, base_url: str) -> httpx.Client:
        return httpx.Client(
            base_url=base_url,
            timeout=self._timeout,
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
        os.environ.setdefault("WEBSOCKET_CLIENT_ENABLE_IPV6", "0")

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
