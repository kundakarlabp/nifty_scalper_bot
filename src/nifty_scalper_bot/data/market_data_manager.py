"""Central market data manager responsible for tick fan-out and broker cache."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from random import uniform
from typing import Any, Callable, Deque, Iterable, Mapping, Sequence, cast

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.data.resolver import InstrumentResolver
from nifty_scalper_bot.data.websocket.manager import ConnectionState, WebSocketManager
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.env import get_str
from nifty_scalper_bot.utils.logging import get_logger, get_tracer_logger
from nifty_scalper_bot.utils.metrics import Counter

TickCallback = Callable[[dict[str, Any]], None]

_EXPIRY_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d-%b-%Y",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S.%f",
    "%d %b %Y",
    "%d %b %Y %H:%M",
    "%d %b %Y %H:%M:%S",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
)

_COMPACT_EXPIRY_FORMATS: tuple[str, ...] = ("%d%b%Y", "%d%b%y")

_logger = get_tracer_logger(__name__)


@dataclass(slots=True)
class _OHLCBar:
    """Normalized one-minute OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class _OHLCBuilder:
    """Aggregate ticks into fixed one-minute OHLC bars."""

    def __init__(self, *, maxlen: int = 500) -> None:
        self._bars: dict[str, Deque[_OHLCBar]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._last_cumulative_volume: dict[str, float] = {}
        self._lock = threading.RLock()

    def add_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Update the latest bar for *symbol* with the provided tick."""

        if not symbol:
            return
        ts_value = tick.get("timestamp")
        price_value = tick.get("ltp")
        if not isinstance(ts_value, (int, float)) or not isinstance(
            price_value, (int, float)
        ):
            return
        timestamp = datetime.fromtimestamp(float(ts_value), timezone.utc).replace(
            second=0, microsecond=0
        )
        price = float(price_value)
        volume_delta = 0.0
        raw_volume = tick.get("volume")
        if isinstance(raw_volume, (int, float)):
            cumulative = float(raw_volume)
            last_cumulative = self._last_cumulative_volume.get(symbol)
            if last_cumulative is not None:
                diff = cumulative - last_cumulative
                if diff >= 0:
                    volume_delta = diff
            self._last_cumulative_volume[symbol] = cumulative
        with self._lock:
            bucket = self._bars[symbol]
            if bucket and bucket[-1].timestamp == timestamp:
                bar = bucket[-1]
                bar.high = max(bar.high, price)
                bar.low = min(bar.low, price)
                bar.close = price
                bar.volume += volume_delta
            else:
                bucket.append(
                    _OHLCBar(
                        timestamp=timestamp,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=volume_delta,
                    )
                )

    def get_bars(
        self, symbol: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return a snapshot of recent bars for *symbol*."""

        with self._lock:
            bars = list(self._bars.get(symbol, ()))
        if limit is not None and limit >= 0:
            bars = bars[-limit:]
        return [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Return complete bar snapshot for all tracked symbols."""

        with self._lock:
            symbols = list(self._bars.keys())
        return {symbol: self.get_bars(symbol) for symbol in symbols}


class MarketDataManager:
    """Central hub for normalized market data with subscriber fan-out."""

    def __init__(
        self,
        broker_client: Any,
        ws_manager: WebSocketManager | None = None,
        *,
        cache_len: int = 1_000,
        duplicate_window_ms: int = 200,
        resolver: InstrumentResolver | None = None,
    ) -> None:
        self._broker = broker_client
        self._ws = ws_manager
        self._cache_len = cache_len
        self._duplicate_window = max(duplicate_window_ms, 0) / 1000.0
        self._resolver = resolver
        if self._resolver is not None:
            self._logger.info("Warming up InstrumentResolver cache...")
            self._resolver.warm()
            self._logger.info("InstrumentResolver warmup complete")
        self._logger = get_logger(__name__)

        self._subscribers: dict[str, set[TickCallback]] = defaultdict(set)
        self._latest_ticks: dict[str, dict[str, Any]] = {}
        self._history: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._cache_len)
        )
        self._token_by_symbol: dict[str, int] = {}
        self._symbol_by_token: dict[int, str] = {}
        self._last_signature: dict[
            str, tuple[tuple[float | None, float | None, float | None], float]
        ] = {}
        self._last_tick_wallclock: dict[str, float] = {}
        self._last_quote_ts_ms: dict[str, float] = {}
        self._last_mid: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._account_lock = threading.RLock()
        self._last_tick_source: dict[str, str] = {}
        self._last_tick_hash: dict[str, int] = {}
        self._ws_connected = False
        self._last_hb_mono: float | None = None
        self._heartbeat_callbacks: list[Callable[[float], None]] = []
        self._fallback_enabled = False
        self._poll_jitter_pct = 0.0
        self._poll_batch_ceiling = 0
        self._ohlc_builder = _OHLCBuilder(maxlen=cache_len)
        self._account_snapshot: dict[str, float] = {}
        self._account_updated_at: float = 0.0
        self._account_cache_ttl = self._parse_float_env(
            "MDM_ACCOUNT_CACHE_TTL", default=30.0, minimum=1.0
        )
        self._account_segment = _resolve_account_segment()

        self._margin_lock = threading.RLock()
        self._margin_snapshot: dict[str, Any] | None = None
        self._last_margin_refresh: float = 0.0
        self._margin_cache_ttl = self._parse_float_env(
            "MDM_MARGIN_TTL_SEC", default=15.0, minimum=1.0
        )
        margin_segment = (
            get_str("ZERODHA_MARGIN_SEGMENT", "BROKER_MARGIN_SEGMENT", default="equity")
            or "equity"
        ).strip().lower()
        if margin_segment not in {"equity", "commodity"}:
            margin_segment = "equity"
        self._margin_segment = margin_segment

        poll_env = os.getenv("MDM_POLL_FALLBACK", "").strip().lower()
        self._rest_poll_enabled = poll_env in {"1", "true", "yes", "on"}
        self._rest_poll_interval = self._parse_float_env(
            "MDM_POLL_INTERVAL_SECONDS", default=3.0, minimum=0.5
        )
        self._rest_poll_max_symbols = self._parse_int_env(
            "MDM_POLL_MAX_SYMBOLS", default=5, minimum=1
        )
        if not self._rest_poll_enabled:
            self._rest_poll_max_symbols = 0
        self._fallback_enabled = self._rest_poll_enabled
        self._rest_poll_stop = threading.Event()
        self._rest_poll_thread: threading.Thread | None = None
        self._tick_stale_threshold_ms = self._parse_int_env(
            "TICK_STALE_MS", default=2_000, minimum=0
        )

        try:
            settings = get_settings()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager.__init__: %s",
                exc,
                extra={"event": "mdm_settings_load_failed"},
            )
            settings = None
        if settings is not None:
            try:
                jitter_pct = max(
                    0.0, float(getattr(settings, "poll_interval_ms_jitter_pct", 0.0))
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarketDataManager.__init__: %s",
                    exc,
                    extra={"event": "mdm_settings_jitter_invalid"},
                )
                jitter_pct = 0.0
            self._poll_jitter_pct = jitter_pct
            try:
                ceiling = int(getattr(settings, "poll_batch_size", 0))
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarketDataManager.__init__: %s",
                    exc,
                    extra={"event": "mdm_settings_batch_invalid"},
                )
                ceiling = 0
            if ceiling > 0:
                self._poll_batch_ceiling = ceiling
                self._rest_poll_max_symbols = min(self._rest_poll_max_symbols, ceiling)

        if self._ws is not None:
            self._ws.on_tick = self._handle_tick
            with suppress(Exception):
                self._ws_connected = bool(self._ws.is_connected())

        self._m_ticks = Counter("mdm_ticks_total", "Normalized ticks processed")
        self._m_subs = Counter("mdm_subscribe_total", "Symbol subscriptions")

    @staticmethod
    def _bar_symbol_key(symbol: str) -> str:
        """Return normalized bar key for *symbol*."""

        return symbol.split(":")[-1].strip().upper()

    def _now_ms(self) -> float:
        """Return the current wall-clock timestamp in milliseconds.

        Returns:
            Floating-point millisecond timestamp.

        Raises:
            None.
        """

        return time.time() * 1000.0

    # ------------------------------------------------------------------
    # Option chain helpers
    def get_option_chain(
        self,
        expiry: str,
        *,
        underlying: str = "NIFTY",
        limit: int = 60,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Return option chain data for the specified *expiry* window."""

        resolver = getattr(self, "_resolver", None)
        normalized_underlying = (underlying or "").strip().upper()
        if not normalized_underlying or resolver is None:
            return None

        try:
            option_contracts = resolver.option_contracts(  # type: ignore[attr-defined]
                normalized_underlying,
                force_refresh=force_refresh,
            )
        except AttributeError:
            return None
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "option_chain_metadata_failed", extra={"error": str(exc)}
            )
            return None

        parsed_contracts: list[dict[str, Any]] = []
        expiries: set[datetime] = set()
        for contract in option_contracts:
            if not isinstance(contract, Mapping):
                continue
            expiry_value = contract.get("expiry")
            expiry_dt = _parse_expiry(expiry_value)
            if expiry_dt is None:
                continue
            instrument_token = contract.get("instrument_token")
            try:
                token = int(instrument_token)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            option_type = str(contract.get("option_type") or "").upper()
            if option_type not in {"CE", "PE"}:
                continue
            tradingsymbol = str(
                contract.get("tradingsymbol") or contract.get("symbol") or ""
            ).strip()
            if not tradingsymbol:
                continue
            strike = _coerce_float(contract.get("strike"))
            parsed_contracts.append(
                {
                    "instrument_token": token,
                    "tradingsymbol": tradingsymbol,
                    "expiry": expiry_dt,
                    "option_type": option_type,
                    "strike": strike,
                    "lot_size": _coerce_int(contract.get("lot_size")),
                    "tick_size": _coerce_float(contract.get("tick_size")),
                    "raw": contract,
                }
            )
            expiries.add(expiry_dt)

        if not parsed_contracts:
            return None

        now = datetime.now(timezone.utc)
        target_expiry = _select_expiry(expiry, sorted(expiries), now)
        if target_expiry is None:
            return None

        target_contracts = [
            contract
            for contract in parsed_contracts
            if contract["expiry"].date() == target_expiry.date()
        ]
        if not target_contracts:
            return None

        underlying_price = self._resolve_underlying_price(normalized_underlying)
        if underlying_price is None:
            strikes = sorted(
                strike
                for strike in (c["strike"] for c in target_contracts)
                if strike is not None
            )
            if strikes:
                underlying_price = strikes[len(strikes) // 2]
        if underlying_price is None:
            return None

        def _contract_sort_key(contract: dict[str, Any]) -> tuple[float, str, float]:
            strike_value = cast(float | None, contract.get("strike"))
            distance = float("inf")
            strike_sort = float("inf")
            if strike_value is not None:
                distance = abs(strike_value - underlying_price)
                strike_sort = strike_value
            option_type_value = cast(str, contract.get("option_type", ""))
            return (distance, option_type_value, strike_sort)

        ranked = sorted(target_contracts, key=_contract_sort_key)
        if limit > 0:
            ranked = ranked[: max(1, int(limit))]

        tokens = [contract["instrument_token"] for contract in ranked]
        quotes = self._fetch_option_quotes(tokens)

        chain: list[dict[str, Any]] = []
        for contract in ranked:
            token = contract["instrument_token"]
            quote = quotes.get(token, {})
            chain.append(_compose_chain_entry(contract, quote))

        chain.sort(
            key=lambda row: (
                row.get("strike", float("inf")),
                row.get("option_type", ""),
            )
        )
        return chain

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def start(self) -> None:
        if self._ws is not None:
            self._ws.start()
        if self._rest_poll_enabled:
            self._start_rest_poll()
        try:
            self.refresh_margin_snapshot(force=True)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager.start margin refresh: %s",
                exc,
                extra={"event": "mdm_start_margin_refresh_error"},
                exc_info=exc,
            )

    def stop(self) -> None:
        if self._ws is not None:
            self._ws.stop()
        if self._rest_poll_thread is not None:
            self._rest_poll_stop.set()
            self._rest_poll_thread.join(timeout=2.0)
            self._rest_poll_thread = None
            self._rest_poll_stop.clear()

    def set_ws_connected(self, connected: bool) -> None:
        """Record WebSocket connectivity state for health reporting."""

        self._ws_connected = bool(connected)

    def register_heartbeat_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback invoked whenever a heartbeat is recorded.

        Args:
            callback: Callable receiving the heartbeat timestamp.

        Returns:
            None.

        Raises:
            ValueError: If *callback* is not callable.
        """

        self._logger.debug(
            "Entered MarketDataManager.register_heartbeat_callback",
            extra={"event": "market_data_register_heartbeat"},
        )
        if not callable(callback):
            self._logger.error(
                (
                    "Failure in MarketDataManager.register_heartbeat_callback: "
                    "invalid callback"
                ),
                extra={"event": "market_data_register_heartbeat_invalid"},
            )
            raise ValueError("callback must be callable")
        with self._lock:
            if callback in self._heartbeat_callbacks:
                self._logger.info(
                    "Condition met: market_data_register_heartbeat_duplicate",
                    extra={"event": "market_data_register_heartbeat_duplicate"},
                )
                return
            self._heartbeat_callbacks.append(callback)

    def bump_heartbeat(self, ts: float | None = None) -> None:
        """Update the last heartbeat timestamp for stale detection.

        Args:
            ts: Optional monotonic timestamp provided by caller.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketDataManager.bump_heartbeat",
            extra={"event": "market_data_bump_heartbeat"},
        )

        self._last_hb_mono = float(ts) if ts is not None else time.monotonic()
        callbacks: list[Callable[[float], None]]
        with self._lock:
            callbacks = list(self._heartbeat_callbacks)
        if not callbacks:
            return
        for callback in callbacks:
            try:
                callback(self._last_hb_mono)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarketDataManager.bump_heartbeat callback: %s",
                    exc,
                    extra={"event": "market_data_heartbeat_callback_error"},
                )

    def heartbeat_age(self) -> float | None:
        """Return the seconds elapsed since the last recorded heartbeat."""

        if self._last_hb_mono is None:
            return None
        return max(0.0, time.monotonic() - self._last_hb_mono)

    # ------------------------------------------------------------------
    # Broker account snapshot accessors
    def get_account_snapshot(self, *, force: bool = False) -> dict[str, float]:
        """Return cached broker margin snapshot sourced via the MDM.

        Args:
            force: Force refresh from the broker when ``True``.

        Returns:
            dict[str, float]: Normalized margin snapshot keyed by broker fields.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketDataManager.get_account_snapshot",
            extra={"event": "mdm_account_snapshot_enter", "force": force},
        )

        now = time.time()
        with self._account_lock:
            cache_age = now - self._account_updated_at
            cache_valid = (
                bool(self._account_snapshot)
                and cache_age < self._account_cache_ttl
                and not force
            )
            if cache_valid:
                self._logger.debug(
                    "Condition met: mdm_account_cache_hit",
                    extra={
                        "event": "mdm_account_cache_hit",
                        "age": round(cache_age, 2),
                        "ttl": self._account_cache_ttl,
                    },
                )
                return dict(self._account_snapshot)

        segment = self._account_segment
        snapshot: dict[str, float] = {}
        response: Any | None = None
        try:
            summary_fetcher = getattr(self._broker, "get_margin_summary", None)
            if callable(summary_fetcher):
                response = summary_fetcher(segment=segment)
                snapshot = _coerce_margin_summary(response)
                if snapshot:
                    self._logger.info(
                        "Condition met: mdm_account_summary_used",
                        extra={
                            "event": "mdm_account_summary_used",
                            "segment": segment,
                        },
                    )

            if not snapshot:
                margins_fetcher = getattr(self._broker, "get_margins", None)
                if callable(margins_fetcher):
                    response = margins_fetcher(segment=segment)
                    normalizer = getattr(
                        self._broker, "_normalize_margin_payload", None
                    )
                    if callable(normalizer):
                        normalized = normalizer(response, segment=segment)
                        snapshot = _coerce_margin_summary(normalized)
                    else:
                        snapshot = _coerce_margin_summary(response)

            if not snapshot:
                balance_fetcher = getattr(self._broker, "get_available_balance", None)
                if callable(balance_fetcher):
                    available = balance_fetcher(segment=segment)
                    available_value = _coerce_positive_float(available)
                    if available_value is not None:
                        snapshot = {"available": float(available_value)}

            if snapshot:
                with self._account_lock:
                    self._account_snapshot = snapshot
                    self._account_updated_at = time.time()
                self._logger.info(
                    "mdm_account_snapshot_updated",
                    extra={
                        "event": "mdm_account_snapshot_updated",
                        "segment": segment,
                        "available": snapshot.get("available"),
                        "net": snapshot.get("net"),
                        "used": snapshot.get("used"),
                    },
                )
                return dict(snapshot)

            self._logger.warning(
                "mdm_account_snapshot_empty",
                extra={"event": "mdm_account_snapshot_empty", "segment": segment},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager.get_account_snapshot: %s",
                exc,
                extra={"event": "mdm_account_snapshot_error", "segment": segment},
                exc_info=exc,
            )

        with self._account_lock:
            fallback = dict(self._account_snapshot)
        if fallback:
            self._logger.info(
                "Condition met: mdm_account_snapshot_fallback",
                extra={
                    "event": "mdm_account_snapshot_fallback",
                    "segment": segment,
                    "available": fallback.get("available"),
                },
            )
        return fallback

    def get_available_balance(self, *, force: bool = False) -> float | None:
        """Return latest available margin balance from cached snapshot.

        Args:
            force: Force refresh of the underlying account snapshot when ``True``.

        Returns:
            float | None: Positive available margin balance when present.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketDataManager.get_available_balance",
            extra={"event": "mdm_available_balance_enter", "force": force},
        )
        margin_snapshot: dict[str, Any] | None = None
        try:
            if force:
                margin_snapshot = self.refresh_margin_snapshot(force=True)
            else:
                margin_snapshot = self.get_margin_snapshot()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager.get_available_balance margin: %s",
                exc,
                extra={"event": "mdm_available_balance_margin_error"},
                exc_info=exc,
            )
            margin_snapshot = None

        if margin_snapshot:
            margin_value = margin_snapshot.get("available")
            numeric_margin = _coerce_float(margin_value)
            if (
                numeric_margin is not None
                and math.isfinite(numeric_margin)
                and numeric_margin >= 0
            ):
                return float(numeric_margin)

        snapshot = self.get_account_snapshot(force=force)
        if not snapshot:
            self._logger.warning(
                "mdm_available_balance_missing_snapshot",
                extra={"event": "mdm_available_balance_missing_snapshot"},
            )
            return None

        candidate_keys = (
            "available",
            "live_balance",
            "cash",
            "opening_balance",
            "net",
        )
        for key in candidate_keys:
            value = snapshot.get(key)
            numeric = _coerce_positive_float(value)
            if numeric is not None:
                self._logger.info(
                    "mdm_available_balance_resolved",
                    extra={
                        "event": "mdm_available_balance_resolved",
                        "key": key,
                        "balance": round(numeric, 2),
                    },
                )
                return float(numeric)

        self._logger.warning(
            "mdm_available_balance_unavailable",
            extra={"event": "mdm_available_balance_unavailable"},
        )
        return None

    # ------------------------------------------------------------------
    # Public API
    def subscribe(self, symbol: str, callback: TickCallback) -> None:
        """Subscribe *callback* to receive normalized ticks for *symbol*."""

        with self._lock:
            subscribers = self._subscribers[symbol]
            subscribers.add(callback)
            latest = self._latest_ticks.get(symbol)
            cached_token = self._token_by_symbol.get(symbol)

        if latest is not None:
            try:
                callback(dict(latest))
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "Warm push callback failed",
                    extra={"symbol": symbol, "error": str(exc)},
                )

        token: int | None = cached_token
        if token is None and self._resolver is not None:
            try:
                token = self._resolver.resolve(symbol)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "Resolver lookup failed",
                    extra={"symbol": symbol, "error": str(exc)},
                )
        self._seed_mapping(symbol, token)

        self._ensure_subscription(symbol)
        try:
            self._m_subs.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass

        resolved_token: int | None = None
        resolver = getattr(self, "_resolver", None)
        if resolver is not None:
            for attr in ("resolve_token", "resolve"):
                if not hasattr(resolver, attr):
                    continue
                try:
                    candidate = getattr(resolver, attr)(symbol)
                except Exception:  # noqa: BLE001
                    candidate = None
                if candidate is not None:
                    resolved_token = candidate
                    break
        if resolved_token is None:
            with self._lock:
                resolved_token = self._token_by_symbol.get(symbol)
        self._seed_mapping(symbol, resolved_token)

    def unsubscribe(self, symbol: str, callback: TickCallback) -> None:
        """Remove *callback* from subscribers of *symbol*."""

        should_unsubscribe = False
        with self._lock:
            callbacks = self._subscribers.get(symbol)
            if callbacks is None:
                return
            callbacks.discard(callback)
            if not callbacks:
                self._subscribers.pop(symbol, None)
                should_unsubscribe = True

        if should_unsubscribe:
            self._release_subscription(symbol)

    def get_latest_tick(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            tick = self._latest_ticks.get(symbol)
            return None if tick is None else dict(tick)

    def get_latest_price(self, symbol: str) -> float | None:
        tick = self.get_latest_tick(symbol)
        if tick is None:
            return None
        try:
            return float(tick["ltp"])
        except (KeyError, TypeError, ValueError):
            return None

    def _resolve_underlying_price(self, symbol: str) -> float | None:
        tick_price = self.get_latest_price(symbol)
        if tick_price is not None:
            return tick_price
        broker = getattr(self, "_broker", None)
        if broker is None or not hasattr(broker, "get_quote"):
            return None
        try:
            quote = broker.get_quote(symbol)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(quote, Mapping):
            price_value = _coerce_float(
                quote.get("ltp")
                or quote.get("last_price")
                or quote.get("close")
                or quote.get("price")
            )
            if price_value is not None:
                return price_value
        return None

    def _fetch_option_quotes(self, tokens: Sequence[int]) -> dict[int, dict[str, Any]]:
        if not tokens:
            return {}
        broker = getattr(self, "_broker", None)
        results: dict[int, dict[str, Any]] = {}
        if broker is None:
            return results
        if hasattr(broker, "get_quote_bulk"):
            try:
                payload = broker.get_quote_bulk(list(tokens))
            except Exception:  # noqa: BLE001
                payload = {}
            if isinstance(payload, Mapping):
                for key, value in payload.items():
                    token: int | None = None
                    if isinstance(key, (int, float)):
                        token = int(key)
                    elif isinstance(key, str):
                        key_candidate = key.strip()
                        if key_candidate:
                            try:
                                token = int(float(key_candidate))
                            except (TypeError, ValueError):
                                token = None
                    if token is None and isinstance(value, Mapping):
                        token = _coerce_int(value.get("instrument_token"))
                    if token is None or token <= 0:
                        continue
                    if isinstance(value, Mapping):
                        results[token] = dict(value)
        missing = [token for token in tokens if token not in results]
        if missing and hasattr(broker, "get_quote_by_token"):
            for token in missing:
                try:
                    quote = broker.get_quote_by_token(int(token))
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(quote, Mapping):
                    results[int(token)] = dict(quote)
        return results

    def get_tick_history(self, symbol: str) -> list[dict[str, Any]]:
        with self._lock:
            history = self._history.get(symbol)
            if history is None:
                return []
            return [dict(item) for item in history]

    def wait_for_symbol(
        self,
        symbol: str,
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> bool:
        """Block until *symbol* has a cached tick or timeout.

        Parameters
        ----------
        symbol:
            Trading symbol to wait for.
        timeout:
            Maximum number of seconds to wait for a cached tick.
        poll_interval:
            Sleep interval between cache checks. Clamped to a sensible minimum
            to avoid busy looping.

        Returns
        -------
        bool
            ``True`` if the symbol already had or received a tick before the
            timeout elapsed, ``False`` otherwise.
        """

        if self.get_latest_tick(symbol) is not None:
            return True

        deadline = time.monotonic() + max(timeout, 0.0)
        interval = max(poll_interval, 0.01)

        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(interval, max(deadline - now, 0.0)))
            if self.get_latest_tick(symbol) is not None:
                return True

        return False

    def pull_quote(self, symbol: str) -> dict[str, Any]:
        quote = self._broker.get_quote(symbol)
        if not isinstance(quote, dict):
            return {"symbol": symbol}
        with self._lock:
            previous = self._latest_ticks.get(symbol)

        normalized = self._normalize_tick(symbol, quote, previous)
        if normalized is not None:
            if not self._is_duplicate(symbol, normalized):
                self._emit_tick(symbol, normalized, source="rest")
            else:
                self._store_tick(symbol, normalized)
        return normalized or {"symbol": symbol, **quote}

    def stats(self) -> dict[str, Any]:
        """Return health metrics for observability."""

        return self.mdm_status()

    def mdm_status(self) -> dict[str, Any]:
        """Return an enriched status snapshot for health and telemetry."""

        with self._lock:
            now = time.time()
            subscriptions = {
                symbol: len(callbacks)
                for symbol, callbacks in self._subscribers.items()
            }
            last_tick_age = {
                symbol: (now - ts if ts is not None else None)
                for symbol, ts in self._last_tick_wallclock.items()
            }
            last_source = dict(self._last_tick_source)

        heartbeat_age = self.heartbeat_age()

        margin_snapshot = self.get_margin_snapshot()
        margin_status: dict[str, Any] = {
            "segment": self._margin_segment,
            "available": None,
            "used": None,
            "net": None,
            "age": None,
        }
        if margin_snapshot is not None:
            margin_status["available"] = margin_snapshot.get("available")
            margin_status["used"] = margin_snapshot.get("used")
            margin_status["net"] = margin_snapshot.get("net")
            fetched_at = margin_snapshot.get("fetched_at")
            if isinstance(fetched_at, (int, float)):
                margin_status["age"] = max(now - float(fetched_at), 0.0)

        return {
            "subscriptions": subscriptions,
            "symbols": len(subscriptions),
            "last_tick_age": last_tick_age,
            "last_tick_source": last_source,
            "ws_connected": self._ws_connected,
            "heartbeat_age": heartbeat_age,
            "fallback_enabled": self._fallback_enabled,
            "tick_stale_threshold_ms": self._tick_stale_threshold_ms,
            "margin": margin_status,
        }

    def refresh_margin_snapshot(self, *, force: bool = False) -> dict[str, Any] | None:
        """Fetch and cache the latest broker margin snapshot.

        Args:
            force: Force a broker fetch regardless of cache staleness.

        Returns:
            dict[str, Any] | None: Copy of the cached margin snapshot.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered refresh_margin_snapshot",
            extra={
                "event": "mdm_margin_refresh_enter",
                "force": force,
                "segment": self._margin_segment,
            },
        )

        now = time.time()
        with self._margin_lock:
            snapshot = self._margin_snapshot
            if (
                not force
                and snapshot is not None
                and now - self._last_margin_refresh < self._margin_cache_ttl
            ):
                self._logger.debug(
                    "Condition met: mdm_margin_cache_hit",
                    extra={"event": "mdm_margin_cache_hit"},
                )
                return dict(snapshot)

        try:
            payload = self._broker_margin_payload()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in refresh_margin_snapshot fetch: %s",
                exc,
                extra={
                    "event": "mdm_margin_refresh_error",
                    "segment": self._margin_segment,
                },
                exc_info=exc,
            )
            with self._margin_lock:
                if self._margin_snapshot is not None:
                    return dict(self._margin_snapshot)
            return None

        normalized = self._normalize_margin_payload(payload)
        if normalized is None:
            self._logger.warning(
                "Condition met: mdm_margin_refresh_no_data",
                extra={"event": "mdm_margin_refresh_no_data"},
            )
            with self._margin_lock:
                if self._margin_snapshot is not None:
                    return dict(self._margin_snapshot)
            return None

        primary = normalized["primary"]
        entries = normalized["entries"]

        snapshot_payload: dict[str, Any] = {
            "segment": primary.get("segment", self._margin_segment),
            "available": primary.get("available"),
            "used": primary.get("used"),
            "net": primary.get("net"),
            "entries": entries,
            "raw": payload,
            "fetched_at": now,
            "source": "broker",
        }

        with self._margin_lock:
            self._margin_snapshot = snapshot_payload
            self._last_margin_refresh = now

        available_value = snapshot_payload.get("available")
        try:
            available_repr = (
                round(float(available_value), 2)
                if isinstance(available_value, (int, float))
                else None
            )
        except Exception:  # pragma: no cover - defensive rounding
            available_repr = None

        self._logger.info(
            "Condition met: mdm_margin_refresh",
            extra={
                "event": "mdm_margin_refresh",
                "segment": snapshot_payload.get("segment"),
                "available": available_repr,
                "entries": len(entries),
            },
        )

        return dict(snapshot_payload)

    def get_margin_snapshot(self) -> dict[str, Any] | None:
        """Return the cached broker margin snapshot.

        Args:
            None.

        Returns:
            dict[str, Any] | None: Copy of cached margin snapshot when available.

        Raises:
            None.
        """

        with self._margin_lock:
            if self._margin_snapshot is None:
                return None
            return dict(self._margin_snapshot)

    def _broker_margin_payload(self) -> Any:
        """Fetch the raw broker margin payload for the configured segment.

        Args:
            None.

        Returns:
            Any: Raw payload returned by the broker margin endpoint.

        Raises:
            None.
        """

        segment = self._margin_segment
        broker = self._broker
        fetchers = (
            "get_account_margins",
            "get_margin_summary",
            "get_margins",
            "margin",
        )
        for name in fetchers:
            fetcher = getattr(broker, name, None)
            if not callable(fetcher):
                continue
            try:
                if name in {"get_margin_summary", "get_margins"}:
                    response = fetcher(segment=segment)
                else:
                    response = fetcher()
            except TypeError:
                try:
                    response = fetcher()
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in _broker_margin_payload (%s): %s",
                        name,
                        exc,
                        extra={
                            "event": "mdm_margin_fetch_error",
                            "method": name,
                            "segment": segment,
                        },
                        exc_info=exc,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _broker_margin_payload (%s): %s",
                    name,
                    exc,
                    extra={
                        "event": "mdm_margin_fetch_error",
                        "method": name,
                        "segment": segment,
                    },
                    exc_info=exc,
                )
                continue
            if response is not None:
                return response
        self._logger.warning(
            "Condition met: mdm_margin_fetch_unsupported",
            extra={
                "event": "mdm_margin_fetch_unsupported",
                "segment": segment,
            },
        )
        return None

    def _normalize_margin_payload(self, payload: Any) -> dict[str, Any] | None:
        """Normalize broker payload into structured margin entries.

        Args:
            payload: Raw payload returned by the broker margin endpoint.

        Returns:
            dict[str, Any] | None: Normalized margin representation.

        Raises:
            None.
        """

        entries = self._collect_margin_entries(payload)
        aggregated: dict[str, dict[str, Any]] = {}
        for segment_hint, raw_entry in entries:
            segment = segment_hint or self._margin_segment
            if not isinstance(segment, str) or not segment.strip():
                segment = self._margin_segment
            normalized_segment = segment.strip().lower()
            available = self._extract_numeric(
                raw_entry,
                ("AVAILABLE", "CASH", "FREE", "BALANCE", "OPENING", "LIVE"),
            )
            used = self._extract_numeric(
                raw_entry,
                ("USED", "UTILISED", "UTILIZED", "BLOCKED"),
            )
            net = self._extract_numeric(
                raw_entry,
                ("NET", "TOTAL", "EQUITY", "LIVE_BALANCE"),
            )
            raw_copy: Any
            if isinstance(raw_entry, Mapping):
                raw_copy = dict(raw_entry)
            else:
                raw_copy = raw_entry
            normalized_entry = {
                "segment": normalized_segment,
                "available": available,
                "used": used,
                "net": net if net is not None else available,
                "raw": raw_copy,
            }
            current = aggregated.get(normalized_segment)
            if current is None:
                aggregated[normalized_segment] = normalized_entry
                continue
            current_available = current.get("available")
            if current_available is None and available is not None:
                aggregated[normalized_segment] = normalized_entry

        if not aggregated:
            fallback_value = self._coerce_positive_float(payload)
            if fallback_value is None:
                return None
            aggregated[self._margin_segment] = {
                "segment": self._margin_segment,
                "available": fallback_value,
                "used": None,
                "net": fallback_value,
                "raw": payload,
            }

        entries_list = list(aggregated.values())
        primary = aggregated.get(self._margin_segment)
        if primary is None and entries_list:
            primary = entries_list[0]
        if primary is None:
            return None
        return {"primary": primary, "entries": entries_list}

    def _collect_margin_entries(self, payload: Any) -> list[tuple[str | None, Any]]:
        """Collect potential segment entries from raw payload.

        Args:
            payload: Raw broker payload to inspect.

        Returns:
            list[tuple[str | None, Mapping[str, Any] | Any]]: Candidate entries.

        Raises:
            None.
        """

        entries: list[tuple[str | None, Mapping[str, Any] | Any]] = []
        stack: list[Any] = [payload]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                marker = id(current)
                if marker in seen:
                    continue
                seen.add(marker)
                segment_value = current.get("segment")
                segment_hint: str | None = None
                if isinstance(segment_value, str) and segment_value.strip():
                    segment_hint = segment_value.strip().lower()
                    entries.append((segment_hint, current))
                keywords = {"equity", "commodity"}
                for key, value in current.items():
                    if isinstance(key, str) and key.strip().lower() in keywords:
                        normalized_key = key.strip().lower()
                        if isinstance(value, Mapping):
                            entries.append((normalized_key, value))
                        elif isinstance(value, (int, float)):
                            entries.append(
                                (
                                    normalized_key,
                                    {"available": value, "segment": normalized_key},
                                )
                            )
                    if isinstance(value, (Mapping, list, tuple, set)):
                        stack.append(value)
                if segment_hint is None:
                    entries.append((None, current))
            elif isinstance(current, (list, tuple, set)):
                for item in current:
                    stack.append(item)
        return entries

    def _extract_numeric(self, payload: Any, keywords: tuple[str, ...]) -> float | None:
        """Extract a numeric field from nested payload using keyword hints.

        Args:
            payload: Nested payload to inspect.
            keywords: Keyword tokens considered indicative of the target value.

        Returns:
            float | None: Positive numeric value when discovered.

        Raises:
            None.
        """

        stack: list[Any] = [payload]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                marker = id(current)
                if marker in seen:
                    continue
                seen.add(marker)
                for key, value in current.items():
                    if isinstance(key, str):
                        upper = key.upper()
                        if any(token in upper for token in keywords):
                            number = self._coerce_positive_float(value)
                            if number is not None:
                                return number
                    if isinstance(value, (Mapping, list, tuple, set)):
                        stack.append(value)
                    else:
                        number = self._coerce_positive_float(value)
                        if number is not None and not keywords:
                            return number
            elif isinstance(current, (list, tuple, set)):
                for item in current:
                    stack.append(item)
            else:
                number = self._coerce_positive_float(current)
                if number is not None and not keywords:
                    return number
        return None

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        """Coerce *value* into a finite non-negative float when possible.

        Args:
            value: Input value to convert.

        Returns:
            float | None: Converted positive float or ``None`` when invalid.

        Raises:
            None.
        """

        if isinstance(value, (int, float)):
            number = float(value)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
        if not math.isfinite(number):
            return None
        if number < 0:
            return None
        return number

    @property
    def ws_connected(self) -> bool:
        """Expose WebSocket connectivity for compatibility."""

        return self._ws_connected

    def is_live(self) -> bool:
        if self._ws is None:
            return True
        state = self._ws.connection_state()
        if state != ConnectionState.CONNECTED:
            return self._rest_poll_enabled and self._has_recent_rest_ticks()
        if not self._ws.is_connected():
            return self._rest_poll_enabled and self._has_recent_rest_ticks()
        return True

    # ------------------------------------------------------------------
    # Internal plumbing
    def _seed_mapping(self, symbol: str, token: int | None) -> None:
        if token is None:
            return
        try:
            token_int = int(token)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._token_by_symbol[symbol] = token_int
            self._symbol_by_token[token_int] = symbol

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        raw_token = tick.get("instrument_token")
        if raw_token is None:
            raw_token = tick.get("token")
        token: int | None
        if raw_token is None:
            token = None
        else:
            try:
                token = int(raw_token)
            except (TypeError, ValueError):
                token = None
        symbol = self._symbol_by_token.get(token) if token is not None else None
        if symbol is None:
            symbol = self._extract_symbol(tick)
            if symbol and token is not None:
                self._seed_mapping(symbol, token)
        if not symbol:
            self._logger.debug("Dropping tick without symbol", extra={"raw": tick})
            return

        raw_ltp = tick.get("last_price")
        if raw_ltp is None:
            raw_ltp = tick.get("ltp")
        try:
            ltp = float(raw_ltp) if raw_ltp is not None else 0.0
        except (TypeError, ValueError):
            ltp = 0.0

        raw_timestamp = tick.get("timestamp")
        if isinstance(raw_timestamp, datetime):
            timestamp_value = raw_timestamp.timestamp()
        elif isinstance(raw_timestamp, (int, float)):
            timestamp_value = float(raw_timestamp)
        elif isinstance(raw_timestamp, str):
            try:
                timestamp_value = float(raw_timestamp)
            except ValueError:
                timestamp_value = time.time()
        else:
            timestamp_value = time.time()
        tick_hash = hash((ltp, int(timestamp_value * 1000)))

        duplicate_detected = False
        with self._lock:
            previous = self._latest_ticks.get(symbol)
            last_hash = self._last_tick_hash.get(symbol)
            if last_hash == tick_hash:
                duplicate_detected = True
            else:
                self._last_tick_hash[symbol] = tick_hash

        if duplicate_detected:
            self._logger.debug(
                "Duplicate tick ignored",
                extra={
                    "event": "market.tick.dedupe",
                    "symbol": symbol,
                    "ltp": ltp,
                },
            )
            return

        normalized = self._normalize_tick(symbol, tick, previous)
        if normalized is None:
            return

        if previous is None:
            self._logger.info(
                "mdm_first_tick_cached",
                extra={
                    "symbol": symbol,
                    "source": normalized.get("source"),
                },
            )

        stale_threshold = self._tick_stale_threshold_ms
        if stale_threshold > 0:
            ts_value = normalized.get("timestamp")
            if isinstance(ts_value, (int, float)):
                age_ms = max(0.0, (time.time() - float(ts_value)) * 1000.0)
                if age_ms > stale_threshold:
                    self._logger.warning(
                        "Stale tick dropped",  # pragma: no cover - logging path
                        extra={
                            "symbol": symbol,
                            "age_ms": round(age_ms, 2),
                            "threshold_ms": stale_threshold,
                        },
                    )
                    return

        loop_now = time.monotonic()
        if self._is_duplicate(symbol, normalized, now=loop_now):
            return

        source = "ws"
        if self._ws is None:
            source = "rest"
        else:
            self.set_ws_connected(True)
        self.bump_heartbeat(loop_now)
        self._emit_tick(symbol, normalized, source=source)

    def _store_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        """Persist normalized *tick* for *symbol* and refresh derived series."""

        wallclock = tick.get("timestamp", time.time())
        cached_tick = dict(tick)
        with self._lock:
            self._latest_ticks[symbol] = cached_tick
            self._history[symbol].append(cached_tick)
            self._last_tick_wallclock[symbol] = float(wallclock)
        bar_symbol = self._bar_symbol_key(symbol)
        self._ohlc_builder.add_tick(bar_symbol, cached_tick)
        staleness_seconds = 0.0
        try:
            staleness_seconds = max(time.time() - float(wallclock), 0.0)
        except Exception:  # pragma: no cover - defensive fallback
            staleness_seconds = 0.0
        try:
            METRICS.observe_tick(
                symbol=symbol,
                latency_seconds=None,
                staleness_seconds=staleness_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager._store_tick: %s",
                exc,
                extra={
                    "event": "market_data_staleness_metric_error",
                    "symbol": symbol,
                },
            )

    def _emit_tick(self, symbol: str, tick: dict[str, Any], *, source: str) -> None:
        self._store_tick(symbol, tick)
        callbacks: list[TickCallback]
        with self._lock:
            self._last_tick_source[symbol] = source
            callbacks = list(self._subscribers.get(symbol, ()))
        if source != "ws":
            self.bump_heartbeat()
        try:
            self._m_ticks.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        for callback in callbacks:
            try:
                callback(dict(tick))
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Tick callback failed",
                    extra={"symbol": symbol, "error": str(exc)},
                )

    def _start_rest_poll(self) -> None:
        if self._rest_poll_thread is not None and self._rest_poll_thread.is_alive():
            return
        self._rest_poll_stop.clear()
        self._rest_poll_thread = threading.Thread(
            target=self._rest_poll_loop,
            name="mdm-rest-poll",
            daemon=True,
        )
        self._rest_poll_thread.start()

    def _rest_poll_loop(self) -> None:
        """Poll the broker REST API with jittered cadence.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _rest_poll_loop",
            extra={
                "event": "mdm_rest_poll_loop_enter",
                "interval": self._rest_poll_interval,
            },
        )
        base_interval = max(self._rest_poll_interval, 0.1)
        while not self._rest_poll_stop.is_set():
            sleep_for = self._jittered_interval(base_interval)
            if self._rest_poll_stop.wait(sleep_for):
                break
            try:
                self.refresh_margin_snapshot()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarketDataManager._rest_poll_loop margin refresh: %s",
                    exc,
                    extra={"event": "mdm_rest_poll_margin_error"},
                    exc_info=exc,
                )
            symbols = self._symbols_for_poll()
            if not symbols:
                continue
            for symbol in symbols:
                if self._rest_poll_stop.is_set():
                    return
                try:
                    self._poll_symbol(symbol)
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "REST poll failed",
                        extra={"symbol": symbol, "error": str(exc)},
                    )

    def _symbols_for_poll(self) -> list[str]:
        with self._lock:
            candidates = list(self._subscribers.keys())
            if not candidates:
                candidates = list(self._latest_ticks.keys())
        limit = self._rest_poll_max_symbols
        if limit > 0 and len(candidates) > limit:
            candidates = candidates[:limit]
        ceiling = self._poll_batch_ceiling
        if ceiling > 0 and len(candidates) > ceiling:
            return candidates[:ceiling]
        return candidates

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Return the most recent normalized quote for *symbol*.

        Args:
            symbol: Trading symbol identifier.

        Returns:
            Latest cached quote dictionary if present, else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered get_quote",
            extra={"event": "mdm_get_quote_enter", "symbol": symbol},
        )
        try:
            latest = self.get_latest_tick(symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in get_quote: %s",
                exc,
                extra={
                    "event": "mdm_get_quote_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return None
        if latest is None:
            return None
        quote = dict(latest)
        now_ms = self._now_ms()
        bid = _coerce_float(quote.get("bid"))
        ask = _coerce_float(quote.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2.0
            self._last_mid[symbol] = (mid, now_ms)
        self._last_quote_ts_ms[symbol] = now_ms
        return quote

    def get_orderbook(self, symbol: str, *, levels: int = 5) -> dict[str, Any] | None:
        """Return aggregated order book metrics for *symbol*."""

        self._logger.debug(
            "Entered get_orderbook",
            extra={
                "event": "mdm_get_orderbook_enter",
                "symbol": symbol,
                "levels": levels,
            },
        )
        normalized = (symbol or "").strip().upper()
        if not normalized:
            self._logger.info(
                "Condition met: get_orderbook_missing_symbol",
                extra={"event": "mdm_get_orderbook_missing_symbol"},
            )
            return None
        depth_limit = levels if isinstance(levels, int) and levels > 0 else 5
        try:
            quote = self.get_quote(normalized)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in get_orderbook: %s",
                exc,
                extra={
                    "event": "mdm_get_orderbook_failed",
                    "symbol": normalized,
                    "error": str(exc),
                },
            )
            return None
        if not isinstance(quote, Mapping):
            return None
        depth = quote.get("depth")
        if not isinstance(depth, Mapping):
            return None
        buy_levels = _normalize_order_levels(depth.get("buy"), depth_limit)
        sell_levels = _normalize_order_levels(depth.get("sell"), depth_limit)
        if not buy_levels and not sell_levels:
            return None
        best_bid = buy_levels[0]["price"] if buy_levels else None
        best_ask = sell_levels[0]["price"] if sell_levels else None
        spread = None
        if best_bid is not None and best_ask is not None:
            spread = max(0.0, best_ask - best_bid)
        total_bid_qty = sum(level["quantity"] for level in buy_levels)
        total_ask_qty = sum(level["quantity"] for level in sell_levels)
        total_bid_notional = sum(
            level["quantity"] * level["price"] for level in buy_levels
        )
        total_ask_notional = sum(
            level["quantity"] * level["price"] for level in sell_levels
        )
        liquidity_score: float | None = None
        total_liquidity = total_bid_qty + total_ask_qty
        if spread is not None and spread > 0:
            liquidity_score = total_liquidity / spread
        elif total_liquidity > 0:
            liquidity_score = total_liquidity
        mid_price = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2.0
        snapshot = {
            "symbol": normalized,
            "levels": depth_limit,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid": mid_price,
            "buy": buy_levels,
            "sell": sell_levels,
            "total_bid_qty": total_bid_qty,
            "total_ask_qty": total_ask_qty,
            "total_bid_notional": total_bid_notional,
            "total_ask_notional": total_ask_notional,
            "liquidity_score": liquidity_score,
            "timestamp_ms": self._now_ms(),
        }
        return snapshot

    def get_ohlc_bars(
        self, symbol: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return trailing one-minute OHLC bars for *symbol*."""

        bar_symbol = self._bar_symbol_key(symbol)
        return self._ohlc_builder.get_bars(bar_symbol, limit=limit)

    @property
    def market_data(self) -> dict[str, Any]:
        """Return derived market data snapshots such as OHLC bars."""

        snapshot: dict[str, Any] = {}
        for symbol, bars in self._ohlc_builder.snapshot().items():
            snapshot[f"{symbol}_bars"] = bars
        return snapshot

    def _candidate_quote_keys(self, symbol: str) -> list[str | int]:
        """Return ordered broker keys for fetching quotes.

        Args:
            symbol: Trading symbol identifier provided by the caller.

        Returns:
            list[str | int]: Prioritised keys suitable for broker quote APIs.

        Raises:
            None.
        """

        symbols: list[str | int] = []
        normalized = (symbol or "").strip().upper()
        if not normalized:
            return symbols
        symbols.append(normalized)
        if not normalized.startswith(("NFO:", "BSE:", "NSE:")) and normalized.endswith(
            ("CE", "PE")
        ):
            symbols.append(f"NFO:{normalized}")
        resolver = getattr(self, "_resolver", None)
        if resolver is not None:
            try:
                meta = resolver.lookup(normalized)
            except Exception as exc:  # noqa: BLE001
                self._logger.info(
                    "resolver_lookup_failed",
                    extra={"symbol": normalized, "error": str(exc)},
                )
            else:
                if isinstance(meta, Mapping):
                    tradingsymbol = str(meta.get("tradingsymbol") or "").upper()
                    exchange = str(meta.get("exchange") or "NFO").upper()
                    token = meta.get("instrument_token")
                    if tradingsymbol:
                        symbols.append(tradingsymbol)
                        symbols.append(f"{exchange}:{tradingsymbol}")
                    if token is not None:
                        try:
                            symbols.append(int(token))
                        except (TypeError, ValueError):
                            self._logger.info(
                                "resolver_token_invalid",
                                extra={"symbol": normalized, "token": token},
                            )
        seen: set[str | int] = set()
        ordered: list[str | int] = []
        for candidate in symbols:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered

    def _broker_quote_any(self, key: str | int) -> dict[str, Any] | None:
        """Fetch a quote for the provided broker key.

        Args:
            key: Broker symbol or token accepted by the broker client.

        Returns:
            dict[str, Any] | None: Quote dictionary when available.

        Raises:
            None.
        """

        broker = getattr(self, "_broker", None)
        if broker is None:
            return None
        try:
            if isinstance(key, int):
                if hasattr(broker, "get_quote_by_token"):
                    quote = broker.get_quote_by_token(key)
                else:
                    return None
            else:
                quote = broker.get_quote(key)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "Failure in _broker_quote_any: %s",
                exc,
                extra={"event": "mdm_quote_fetch_failed", "key": key},
            )
            return None
        if isinstance(quote, Mapping) and quote:
            return dict(quote)
        return None

    def refresh_quote_now(
        self, symbol: str, *, trace_id: str | None = None
    ) -> dict[str, Any] | None:
        """Force a REST quote fetch and stamp caches immediately.

        Args:
            symbol: Trading symbol identifier.
            trace_id: Optional correlation token used for logging.

        Returns:
            dict[str, Any] | None: Latest quote payload copied into the cache.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered refresh_quote_now",
            extra={
                "event": "mdm_refresh_quote_enter",
                "symbol": symbol,
                "trace_id": trace_id,
            },
        )
        candidates = self._candidate_quote_keys(symbol)
        _logger.info(
            "reference_price_refresh",
            extra={
                "symbol": symbol,
                "age_ms": self.quote_age_ms(symbol) if symbol else None,
                "trace_id": trace_id,
            },
        )
        for key in candidates:
            quote = self._broker_quote_any(key)
            _logger.info(
                "refresh_attempt",
                extra={
                    "symbol": symbol,
                    "key": key,
                    "ok": bool(quote),
                    "trace_id": trace_id,
                },
            )
            if not quote:
                continue
            now_ms = self._now_ms()
            bid = _coerce_float(quote.get("bid"))
            ask = _coerce_float(quote.get("ask"))
            mid: float | None = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            with self._lock:
                self._latest_ticks[symbol] = dict(quote)
                self._last_quote_ts_ms[symbol] = now_ms
                if mid is not None:
                    self._last_mid[symbol] = (mid, now_ms)
            return dict(quote)
        self._logger.error(
            "refresh_quote_failed_all_candidates",
            extra={
                "symbol": symbol,
                "candidates": candidates,
                "trace_id": trace_id,
            },
        )
        return None

    def has_quote(self, symbol: str) -> bool:
        """Return whether cached quote data appears usable.

        Args:
            symbol: Trading symbol identifier.

        Returns:
            bool: ``True`` when cached quotes provide pricing fields.

        Raises:
            None.
        """

        quote = self.get_quote(symbol)
        if not isinstance(quote, Mapping):
            return False
        for key in ("ltp", "price", "last_price", "bid", "ask"):
            if _coerce_float(quote.get(key)) is not None:
                return True
        return False

    def quote_age_ms(self, symbol: str) -> int:
        """Return the age in milliseconds of the cached quote for *symbol*.

        Args:
            symbol: Trading symbol identifier.

        Returns:
            Milliseconds since last quote if known, else a large sentinel value.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered quote_age_ms",
            extra={"event": "mdm_quote_age_enter", "symbol": symbol},
        )
        try:
            with self._lock:
                ts_ms = self._last_quote_ts_ms.get(symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in quote_age_ms: %s",
                exc,
                extra={
                    "event": "mdm_quote_age_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return 1_000_000_000
        if ts_ms is None:
            return 1_000_000_000
        age_ms = int(max(0.0, self._now_ms() - float(ts_ms)))
        return age_ms

    def cached_mid(self, symbol: str) -> tuple[float | None, int | None]:
        """Return cached mid-price and age for *symbol* if available.

        Args:
            symbol: Trading symbol identifier.

        Returns:
            Tuple of mid price and age in milliseconds if cached,
            otherwise ``(None, None)``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered cached_mid",
            extra={"event": "mdm_cached_mid_enter", "symbol": symbol},
        )
        try:
            mid_entry = self._last_mid.get(symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in cached_mid: %s",
                exc,
                extra={
                    "event": "mdm_cached_mid_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return None, None
        if not mid_entry:
            return None, None
        mid_price, ts_ms = mid_entry
        age_ms = int(max(0.0, self._now_ms() - float(ts_ms)))
        return float(mid_price), age_ms

    def _jittered_interval(self, base: float) -> float:
        """Return a jittered interval preserving *base* expectation."""

        jitter_pct = max(self._poll_jitter_pct, 0.0)
        if jitter_pct <= 0.0:
            return base
        lower = max(0.0, 1.0 - jitter_pct)
        upper = 1.0 + jitter_pct
        try:
            interval = base * uniform(lower, upper)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _jittered_interval: %s",
                exc,
                extra={
                    "event": "mdm_jitter_failed",
                    "interval": base,
                    "error": str(exc),
                },
            )
            return base
        return max(interval, 0.01)

    def _poll_symbol(self, symbol: str) -> None:
        quote = self._broker.get_quote(symbol)
        if not isinstance(quote, dict):
            return
        with self._lock:
            previous = self._latest_ticks.get(symbol)
        normalized = self._normalize_tick(symbol, quote, previous)
        if normalized is None:
            return
        if self._is_duplicate(symbol, normalized):
            return
        self._emit_tick(symbol, normalized, source="rest")

    def _has_recent_rest_ticks(self) -> bool:
        cutoff = time.time() - max(self._rest_poll_interval * 2.0, 5.0)
        with self._lock:
            for symbol, ts in self._last_tick_wallclock.items():
                if self._last_tick_source.get(symbol) == "rest" and ts >= cutoff:
                    return True
        return False

    @staticmethod
    def _parse_float_env(var: str, *, default: float, minimum: float) -> float:
        try:
            value = float(os.getenv(var, ""))
        except ValueError:
            return max(default, minimum)
        if value <= 0:
            return max(default, minimum)
        return max(value, minimum)

    @staticmethod
    def _parse_int_env(var: str, *, default: int, minimum: int) -> int:
        raw = os.getenv(var)
        if raw is None:
            return max(default, minimum)
        try:
            value = int(float(raw))
        except ValueError:
            return max(default, minimum)
        return max(value, minimum)

    def _ensure_subscription(self, symbol: str) -> None:
        if self._ws is None:
            return
        try:
            token = self._token_by_symbol.get(symbol)
            if token is None:
                token = self._resolve_token(symbol)
            if token is None:
                self._logger.warning(
                    "WS subscribe skipped (no token)", extra={"symbol": symbol}
                )
                return
            # Zerodha is safest with 'ltp' or 'quote'. Prefer 'ltp' to minimize payload.
            self._logger.info(
                "Subscribing symbol",
                extra={"symbol": symbol, "token": token, "mode": "ltp"},
            )
            self._ws.subscribe_tokens([token], mode="ltp")
        except Exception as exc:  # noqa: BLE001
            # Log both message and details so it shows up even if the logger ignores
            # 'extra'.
            self._logger.warning(
                "Failed to subscribe symbol %s: %s",
                symbol,
                exc,
                extra={"symbol": symbol, "error": str(exc)},
            )

    def _release_subscription(self, symbol: str) -> None:
        if self._ws is None:
            return
        token = self._token_by_symbol.get(symbol)
        if token is None:
            return
        try:
            self._ws.unsubscribe_tokens([token])
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "Unsubscribe failed",
                extra={"symbol": symbol, "error": str(exc)},
            )

    def _resolve_token(self, symbol: str) -> int | None:
        try:
            if hasattr(self._broker, "get_instrument_token"):
                instrument_token = self._broker.get_instrument_token(symbol)
                if instrument_token is not None:
                    token_int = int(instrument_token)
                    with self._lock:
                        self._token_by_symbol[symbol] = token_int
                        self._symbol_by_token[token_int] = symbol
                    return token_int
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "Broker get_instrument_token failed", extra={"error": str(exc)}
            )

        if self._resolver is not None:
            resolved_token = self._resolver.resolve(symbol)
            if resolved_token is not None:
                with self._lock:
                    self._token_by_symbol[symbol] = resolved_token
                    self._symbol_by_token[resolved_token] = symbol
                return resolved_token

        self._logger.warning(
            "Failed to resolve instrument token for %s",
            symbol,
            extra={"symbol": symbol},
        )
        return None

    # ------------------------------------------------------------------
    # Helpers
    @staticmethod
    def _extract_symbol(tick: dict[str, Any]) -> str | None:
        for key in ("symbol", "tradingsymbol", "instrument"):
            value = tick.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _normalize_tick(
        self,
        symbol: str,
        tick: dict[str, Any],
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ltp = self._coerce_float(tick, "ltp", "last_price", "close")
        if ltp is None or ltp <= 0:
            return None
        bid = self._coerce_float(tick, "best_bid", "bid", "buy_price")
        ask = self._coerce_float(tick, "best_ask", "ask", "sell_price")
        if bid is None:
            depth = tick.get("depth", {})
            bid = self._coerce_from_depth(depth, "buy")
        if ask is None:
            depth = tick.get("depth", {})
            ask = self._coerce_from_depth(depth, "sell")
        if previous is not None:
            if bid is None:
                bid = previous.get("bid")
            if ask is None:
                ask = previous.get("ask")

        timestamp = self._coerce_timestamp(tick)

        normalized = {
            "symbol": symbol,
            "ltp": float(ltp),
            "bid": bid,
            "ask": ask,
            "timestamp": timestamp,
        }
        volume = self._coerce_float(
            tick,
            "volume_traded_today",
            "volume_traded",
            "volume",
            "total_traded_volume",
        )
        if volume is None and previous is not None:
            previous_volume = previous.get("volume")
            if isinstance(previous_volume, (int, float)):
                volume = float(previous_volume)
        if volume is not None:
            normalized["volume"] = float(volume)
        return normalized

    @staticmethod
    def _coerce_float(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coerce_from_depth(depth: Any, side: str) -> float | None:
        if not isinstance(depth, dict):
            return None
        entries = depth.get(side)
        if not isinstance(entries, Iterable):
            return None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            price = entry.get("price")
            if price is None:
                continue
            try:
                return float(price)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coerce_timestamp(tick: dict[str, Any]) -> float:
        value = tick.get("timestamp") or tick.get("ts") or tick.get("ts_ms")
        if isinstance(value, (int, float)):
            val = float(value)
            if val > 1_000_000_000_000:
                val /= 1000.0
            return val
        return time.time()

    def _is_duplicate(
        self,
        symbol: str,
        tick: dict[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        signature = (tick.get("ltp"), tick.get("bid"), tick.get("ask"))
        current = float(now) if now is not None else time.monotonic()
        last = self._last_signature.get(symbol)
        if last is None:
            self._last_signature[symbol] = (signature, current)
            return False
        last_signature, last_ts = last
        if signature == last_signature and (current - last_ts) < self._duplicate_window:
            return True
        self._last_signature[symbol] = (signature, current)
        return False


def _compose_chain_entry(
    contract: Mapping[str, Any], quote: Mapping[str, Any]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "symbol": contract.get("tradingsymbol"),
        "tradingsymbol": contract.get("tradingsymbol"),
        "instrument_token": contract.get("instrument_token"),
        "expiry": contract.get("expiry"),
        "option_type": contract.get("option_type"),
        "strike": contract.get("strike"),
        "lot_size": contract.get("lot_size"),
        "tick_size": contract.get("tick_size"),
    }
    ltp = _coerce_float(
        quote.get("last_price")
        or quote.get("ltp")
        or quote.get("close")
        or quote.get("price")
    )
    entry["ltp"] = ltp if ltp is not None else 0.0
    depth = quote.get("depth") if isinstance(quote, Mapping) else None
    if isinstance(depth, Mapping):
        entry["depth"] = {"buy": depth.get("buy", []), "sell": depth.get("sell", [])}
        entry["bid"] = _top_of_book(depth.get("buy"))
        entry["ask"] = _top_of_book(depth.get("sell"))
    else:
        entry["depth"] = {"buy": [], "sell": []}
        entry["bid"] = None
        entry["ask"] = None
    entry["open_interest"] = _coerce_float(
        quote.get("open_interest") or quote.get("oi") or quote.get("oi_day_high")
    )
    entry["trades"] = _coerce_float(
        quote.get("volume_traded")
        or quote.get("volume_traded_today")
        or quote.get("volume")
    )
    raw_contract = contract.get("raw")
    if isinstance(raw_contract, Mapping):
        entry["raw"] = dict(raw_contract)
    elif isinstance(quote, Mapping):
        entry["raw"] = dict(quote)
    else:
        entry["raw"] = {}
    return entry


def _top_of_book(levels: Any) -> float | None:
    if not isinstance(levels, Sequence):
        return None
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        price_value = _coerce_float(
            level.get("price") or level.get("ltp") or level.get("last_price")
        )
        if price_value is not None:
            return price_value
    return None


def _normalize_order_levels(levels: Any, limit: int) -> list[dict[str, float]]:
    """Return sanitized order book levels limited to *limit* entries."""

    normalized: list[dict[str, float]] = []
    if not isinstance(levels, Sequence):
        return normalized
    depth_limit = max(1, int(limit)) if limit > 0 else 1
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        price = _coerce_positive_float(
            level.get("price")
            or level.get("ltp")
            or level.get("last_price")
            or level.get("best_price")
        )
        if price is None:
            continue
        quantity = _coerce_positive_float(
            level.get("quantity")
            or level.get("qty")
            or level.get("volume")
            or level.get("size")
        )
        normalized.append(
            {
                "price": price,
                "quantity": quantity if quantity is not None else 0.0,
            }
        )
        if len(normalized) >= depth_limit:
            break
    return normalized


def _coerce_margin_summary(payload: Any) -> dict[str, float]:
    """Normalize broker margin payload into flattened float mapping.

    Args:
        payload: Candidate payload returned by broker margin endpoints.

    Returns:
        dict[str, float]: Mapping with sanitized numeric margin values.

    Raises:
        None.
    """

    summary: dict[str, float] = {}
    if not isinstance(payload, Mapping):
        return summary
    for key, value in payload.items():
        label = str(key)
        numeric = _coerce_float(value)
        if numeric is None and isinstance(value, Mapping):
            for nested_key in (
                "available",
                "live_balance",
                "cash",
                "opening_balance",
                "net",
                "used",
            ):
                nested_value = _coerce_float(value.get(nested_key))
                if nested_value is not None:
                    numeric = nested_value
                    break
        if numeric is None:
            continue
        summary[label] = float(max(numeric, 0.0))
    return summary


def _resolve_account_segment() -> str:
    """Return sanitized broker margin segment identifier.

    Args:
        None.

    Returns:
        str: Lower-case segment value accepted by broker endpoints.

    Raises:
        None.
    """

    raw_segment = os.getenv("BROKER_MARGIN_SEGMENT", "equity") or "equity"
    segment = raw_segment.strip().lower()
    if segment not in {"equity", "commodity"}:
        segment = "equity"
    return segment


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: Any) -> float | None:
    """Return positive finite float when coercion succeeds.

    Args:
        value: Input coerced to floating-point.

    Returns:
        Positive finite float when available, otherwise ``None``.

    Raises:
        None.
    """

    number = _coerce_float(value)
    if number is None:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return float(number)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_expiry(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc)
            if value.tzinfo
            else value.replace(tzinfo=timezone.utc)
        )
    if isinstance(value, date):
        return datetime.combine(value, datetime.max.time(), tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        sanitized = text.replace("/", "-")
        iso_candidate = (
            sanitized.rstrip("Z") + "+00:00" if sanitized.endswith("Z") else sanitized
        )
        iso_variants = {iso_candidate, sanitized}
        for candidate in iso_variants:
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            return (
                parsed.astimezone(timezone.utc)
                if parsed.tzinfo
                else parsed.replace(tzinfo=timezone.utc)
            )

        normalized_variants = {
            sanitized,
            sanitized.upper(),
            sanitized.lower(),
            sanitized.title(),
            sanitized.replace(" ", ""),
        }
        for variant in normalized_variants:
            for fmt in _EXPIRY_FORMATS:
                try:
                    parsed = datetime.strptime(variant, fmt)
                except ValueError:
                    continue
                return parsed.replace(tzinfo=timezone.utc)
            for fmt in _COMPACT_EXPIRY_FORMATS:
                try:
                    parsed = datetime.strptime(variant.title(), fmt)
                except ValueError:
                    continue
                return parsed.replace(tzinfo=timezone.utc)
        return None
    return None


def _select_expiry(
    expiry: str, expiries: Sequence[datetime], now: datetime
) -> datetime | None:
    if not expiries:
        return None
    mode = (expiry or "weekly").strip().lower()
    explicit = _parse_expiry(expiry) if mode not in {"weekly", "monthly"} else None
    if explicit is not None:
        for candidate in expiries:
            if candidate.date() == explicit.date():
                return candidate
        return explicit

    upcoming = [candidate for candidate in expiries if candidate >= now]
    pool = upcoming or list(expiries)
    if not pool:
        return None
    if mode == "monthly":
        first = pool[0]
        same_month = [
            candidate
            for candidate in pool
            if candidate.year == first.year and candidate.month == first.month
        ]
        return same_month[-1] if same_month else pool[-1]
    return pool[0]


__all__ = ["MarketDataManager"]
