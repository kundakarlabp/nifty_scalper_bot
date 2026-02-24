"""Central market data manager responsible for tick fan-out and broker cache."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import logging
import math
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Iterable,
    Mapping,
    Sequence,
    cast,
)

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.streaming.websocket_manager import (
    ConnectionState,
    WebSocketManager,
)
from nifty_scalper_bot.utils.env import get_str
from nifty_scalper_bot.utils.logging import get_logger, get_tracer_logger, log_throttled
from nifty_scalper_bot.utils.market_hours import is_market_hours_cached
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.symbols import enforce_canonical, normalize_symbol

# NOTE: resolver is attached at runtime by app.py (ctx.market_data_manager._resolver).
# Avoid importing resolver modules here to prevent circular imports or path issues.
if TYPE_CHECKING:
    from nifty_scalper_bot.data.instruments import InstrumentResolver

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
        broker: Any = None,
        websocket: Any = None,
        settings: dict | None = None,
        *,
        cache_len: int = 1000,
        resolver: Any = None,
        **kwargs,
    ) -> None:
        """
        MarketDataManager constructor.
        """
        self._broker = broker
        self._websocket = websocket
        # FIX: Explicitly assign self._ws for internal use
        self._ws = websocket
        self._settings = settings or {}
        self._resolver = resolver
        self._logger = get_logger(__name__)

        # FIX: Initialize cache_len before it is used
        self._cache_len = cache_len

        # FIX: Initialize duplicate window (Missing in your file, causing the crash)
        self._duplicate_window = self._parse_float_env(
            "MDM_DUPLICATE_WINDOW_SEC", default=1.0, minimum=0.0
        )

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
        self._tick_cache: dict[str, dict[str, Any]] = {}
        self._tick_counter = 0
        self._last_tick_log_time = time.monotonic()
        self._last_tick_time: dict[str, float] = {}
        self._tick_bus: Any | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._async_dispatch_drops = 0
        self._last_async_drop_log = time.monotonic()
        self._ws_connected = False
        self._hydration_status: dict[str, str] = {}
        self._last_hb_mono: float | None = None
        self._heartbeat_callbacks: list[Callable[[float], None]] = []
        self._fallback_enabled = False
        self._poll_jitter_pct = 0.0
        self._poll_batch_ceiling = 0
        self._ohlc_builder = _OHLCBuilder(maxlen=self._cache_len)
        self._account_snapshot: dict[str, float] = {}
        self._account_updated_at: float = 0.0
        self._tracked_symbols: set[str] = set()
        self._tick_stats: dict[str, int] = defaultdict(int)
        self._last_tick_stats_log = time.monotonic()
        self._account_cache_ttl = self._parse_float_env(
            "MDM_ACCOUNT_CACHE_TTL", default=30.0, minimum=1.0
        )
        self._account_segment = _resolve_account_segment()

        self._margin_lock = threading.RLock()
        self._margin_snapshot: dict[str, Any] | None = None
        self._last_margin_refresh: float = 0.0
        self.last_tick_time = 0.0
        self._tick_warn_last: dict[str, float] = (
            {}
        )  # ✅ FIX: rate-limit cache-miss warnings
        self._seed_attempt_last: dict[str, float] = {}
        self._seed_completed = False
        self._seeded_symbols: set[str] = set()
        self._margin_cache_ttl = self._parse_float_env(
            "MDM_MARGIN_TTL_SEC", default=15.0, minimum=1.0
        )
        self._unified_manager: Any | None = None
        margin_segment = (
            (
                get_str(
                    "ZERODHA_MARGIN_SEGMENT", "BROKER_MARGIN_SEGMENT", default="equity"
                )
                or "equity"
            )
            .strip()
            .lower()
        )
        if margin_segment not in {"equity", "commodity"}:
            margin_segment = "equity"
        self._margin_segment = margin_segment

        poll_env = os.getenv("MDM_POLL_FALLBACK", "").strip().lower()
        poll_flag_provided = bool(poll_env)
        self._rest_poll_enabled = poll_env in {"1", "true", "yes", "on"}
        self._rest_poll_interval = self._parse_float_env(
            "MDM_POLL_INTERVAL_SECONDS", default=3.0, minimum=0.5
        )
        configured_poll_max = self._parse_int_env(
            "MDM_POLL_MAX_SYMBOLS", default=50, minimum=1
        )
        self._rest_poll_max_symbols = (
            configured_poll_max if self._rest_poll_enabled else 0
        )
        self._fallback_enabled = self._rest_poll_enabled
        if not poll_flag_provided and not self._rest_poll_enabled and self._ws is None:
            self._rest_poll_enabled = True
            self._rest_poll_interval = max(self._rest_poll_interval, 2.0)
            self._rest_poll_max_symbols = max(configured_poll_max, 5)
            self._fallback_enabled = True
            self._logger.info(
                "Condition met: mdm_rest_poll_auto_enabled",
                extra={
                    "event": "mdm_rest_poll_auto_enabled",
                    "interval": float(self._rest_poll_interval),
                    "max_symbols": int(self._rest_poll_max_symbols),
                },
            )
        self._rest_poll_stop = threading.Event()
        self._rest_poll_thread: threading.Thread | None = None
        self._health_monitor_stop = threading.Event()
        self._health_monitor_thread: threading.Thread | None = None
        self._zombie_symbol = "NSE:NIFTY 50"
        self._zombie_tick_threshold_sec = self._parse_float_env(
            "ZOMBIE_TICK_THRESHOLD_SEC", default=60.0, minimum=10.0
        )
        self._zombie_restart_failures = 0
        self._zombie_restart_window = self._parse_float_env(
            "ZOMBIE_RESTART_WINDOW_SEC", default=120.0, minimum=1.0
        )
        self._zombie_restart_limit = self._parse_int_env(
            "ZOMBIE_RESTART_LIMIT", default=3, minimum=1
        )
        self._zombie_breaker_open_until = 0.0
        self._zombie_last_restart_attempt_at: float = 0.0
        self._zombie_restart_cooldown_sec = self._parse_float_env(
            "ZOMBIE_RESTART_COOLDOWN_SEC", default=30.0, minimum=5.0
        )
        self._zombie_stale_logged = False
        self._rest_refresh_inflight: set[str] = set()
        self._tick_stale_threshold_ms = self._parse_int_env(
            "TICK_STALE_MS", default=2_000, minimum=0
        )
        if self._rest_poll_enabled:
            self._tracked_symbols.add("NSE:NIFTY 50")

        # Load optional settings overrides
        if settings is not None:
            try:
                jitter_pct = max(
                    0.0, float(getattr(settings, "poll_interval_ms_jitter_pct", 0.0))
                )
            except Exception:  # noqa: BLE001
                jitter_pct = 0.0
            self._poll_jitter_pct = jitter_pct
            try:
                ceiling = int(getattr(settings, "poll_batch_size", 0))
            except Exception:  # noqa: BLE001
                ceiling = 0
            if ceiling > 0:
                self._poll_batch_ceiling = ceiling
                self._rest_poll_max_symbols = min(self._rest_poll_max_symbols, ceiling)

        if self._ws is not None:
            if hasattr(self._ws, "_on_tick_callback"):
                self._ws._on_tick_callback = self._handle_tick
            else:
                self._ws.on_tick = self._handle_tick
        if self._tick_bus is not None:
            self._tick_bus.subscribe(self._on_tick)
            with suppress(Exception):
                self._ws_connected = bool(self._ws.is_connected())

        self._m_ticks = Counter("mdm_ticks_total", "Normalized ticks processed")
        self._last_balance_log_time = 0.0

    def ingest_rest_quote(self, symbol: str, quote: Mapping[str, Any]) -> None:
        """Commit REST quote payload for ``symbol`` into cache AND emit to subscribers."""
        if not symbol or not isinstance(quote, Mapping):
            return

        try:
            normalized_symbol = symbol.strip().upper()
            if not normalized_symbol:
                return

            # Normalize fields
            ltp = _coerce_float(quote.get("ltp") or quote.get("last_price"))
            bid = _coerce_float(quote.get("bid"))
            ask = _coerce_float(quote.get("ask"))
            timestamp = _coerce_float(
                quote.get("timestamp") or quote.get("server_ts_s")
            )
            volume = _coerce_float(quote.get("volume") or quote.get("volume_traded"))

            payload = {
                "symbol": normalized_symbol,
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "volume": volume,
                "timestamp": timestamp or time.time(),
                "_source": "rest",
                "depth": quote.get("depth"),
            }

            with self._lock:
                previous = self._latest_ticks.get(normalized_symbol)
                self._latest_ticks[normalized_symbol] = payload

            # [FIX] CRITICAL: Emit to Strategy Runner!
            # [FIX] CRITICAL: Emit to Strategy Runner!
            normalized_tick = self._normalize_tick(normalized_symbol, payload, previous)

            # Fallback: If normalization is too strict but we successfully extracted an LTP
            # in the payload above, use the payload directly.
            tick_to_emit = normalized_tick
            if tick_to_emit is None and payload.get("ltp") is not None:
                tick_to_emit = payload

            if tick_to_emit:
                self._emit_tick(normalized_symbol, tick_to_emit, source="rest")

        except Exception as exc:
            self._logger.error(
                "Failure in MarketDataManager.ingest_rest_quote: %s",
                exc,
                extra={"event": "mdm_ingest_rest_quote_error", "symbol": symbol},
                exc_info=exc,
            )

    def set_unified_manager(self, manager: Any | None) -> None:
        """Attach unified manager callbacks for cache updates.

        Args:
            manager: Unified manager instance to notify or ``None`` to detach.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketDataManager.set_unified_manager",
            extra={"event": "mdm_unified_manager_set_enter"},
        )
        self._unified_manager = manager

    def _notify_unified_manager(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Forward cache updates to the unified manager when configured.

        Args:
            symbol: Symbol identifier associated with the tick payload.
            tick: Mapping of normalized tick fields to forward.

        Returns:
            None.

        Raises:
            None.
        """

        manager = self._unified_manager
        if manager is None or not symbol:
            return
        callback = getattr(manager, "on_cache_update", None)
        if not callable(callback):
            return
        try:
            callback(symbol, dict(tick))
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager._notify_unified_manager: %s",
                exc,
                extra={
                    "event": "mdm_unified_manager_notify_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )

    @staticmethod
    def _bar_symbol_key(symbol: str) -> str:
        """Return normalized bar key for *symbol*."""

        return enforce_canonical(normalize_symbol(symbol))

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
            self._logger.error(
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
            try:
                self._ws.start()
            except Exception:
                self._logger.exception("Failure in ws.start")

        # 🔥 Force REST fallback if WS not connected
        if self._rest_poll_enabled:
            if not self._ws or not self._ws.is_connected():
                self._logger.info(
                    "mdm_rest_fallback_activated",
                    extra={"event": "mdm_rest_fallback_activated"},
                )
                self._start_rest_poll()
        self._start_health_monitor()

    def stop(self) -> None:
        if self._ws is not None:
            self._ws.stop()
        if self._rest_poll_thread is not None:
            self._rest_poll_stop.set()
            self._rest_poll_thread.join(timeout=2.0)
            self._rest_poll_thread = None
            self._rest_poll_stop.clear()
        if self._health_monitor_thread is not None:
            self._health_monitor_stop.set()
            self._health_monitor_thread.join(timeout=2.0)
            self._health_monitor_thread = None
            self._health_monitor_stop.clear()

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
    async def get_account_snapshot(self, *, force: bool = False) -> dict[str, float]:
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
                response = await summary_fetcher(segment=segment)
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
                    response = await margins_fetcher(segment=segment)
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
                    available = await balance_fetcher(segment=segment)
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

            self._logger.error(
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
        try:
            broker = getattr(self, "_broker_client", None) or getattr(
                self, "_broker", None
            )
            if broker is not None:
                segment = _resolve_account_segment()
                if hasattr(broker, "get_available_balance"):
                    try:
                        live_balance = broker.get_available_balance(segment=segment)  # type: ignore[call-arg]
                        numeric_live = (
                            _coerce_positive_float(live_balance)
                            if live_balance is not None
                            else None
                        )
                        if numeric_live is not None:
                            resolved_balance = float(numeric_live)
                            # [FIX] Throttled INFO log
                            if time.time() - self._last_balance_log_time >= 60.0:
                                self._logger.debug(
                                    "mdm_available_balance_resolved",
                                    extra={
                                        "event": "mdm_available_balance_resolved",
                                        "key": "available",
                                        "balance": round(resolved_balance, 2),
                                        "source": "broker_available",
                                    },
                                )
                                self._last_balance_log_time = time.time()
                            return resolved_balance
                    except Exception as exc:  # noqa: BLE001
                        self._logger.error(
                            "mdm_available_balance_direct_error: %s",
                            exc,
                            extra={"event": "mdm_available_balance_direct_error"},
                            exc_info=exc,
                        )

                if hasattr(broker, "get_margin_summary"):
                    summary = broker.get_margin_summary(segment=segment)  # type: ignore[call-arg]
                    if isinstance(summary, Mapping):
                        flattened = _coerce_margin_summary(summary)
                        for key in ("available", "available_cash", "cash", "net"):
                            value = _coerce_positive_float(flattened.get(key))
                            if value is not None:
                                if time.time() - self._last_balance_log_time >= 60.0:
                                    self._logger.debug(
                                        "mdm_available_balance_resolved",
                                        extra={
                                            "event": "mdm_available_balance_resolved",
                                            "key": key,
                                            "balance": round(value, 2),
                                            "source": "broker_margin",
                                        },
                                    )
                                    self._last_balance_log_time = time.time()
                                return float(value)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "mdm_available_balance_margin_error: %s",
                exc,
                extra={"event": "mdm_available_balance_margin_error"},
                exc_info=exc,
            )

        snapshot = self.get_account_snapshot(force=force)
        if not isinstance(snapshot, Mapping):
            self._logger.error(
                "mdm_available_balance_missing_snapshot",
                extra={"event": "mdm_available_balance_missing_snapshot"},
            )
            return None

        candidate_keys = (
            "available",
            "available_cash",
            "live_balance",
            "cash",
            "net",
        )
        for key in candidate_keys:
            value = snapshot.get(key)
            numeric = _coerce_positive_float(value)
            if numeric is not None:
                self._logger.debug(
                    "mdm_available_balance_resolved",
                    extra={
                        "event": "mdm_available_balance_resolved",
                        "key": key,
                        "balance": round(numeric, 2),
                        "source": "account_snapshot",
                    },
                )
                return float(numeric)

        self._logger.error(
            "mdm_available_balance_unavailable",
            extra={"event": "mdm_available_balance_unavailable"},
        )
        return None

    # ------------------------------------------------------------------
    # Public API
    def subscribe(self, symbol: str, callback: TickCallback) -> None:
        """Subscribe *callback* to receive normalized ticks for *symbol*."""

        symbol = enforce_canonical(normalize_symbol(str(symbol)))
        if symbol.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {symbol}")
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

        symbol = enforce_canonical(normalize_symbol(str(symbol)))
        if symbol.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {symbol}")
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

    def get_latest_tick(self, symbol: str | int) -> dict[str, Any] | None:
        """Args: symbol; Returns: cached tick snapshot; Raises: none."""
        resolved_symbol: str | None
        if isinstance(symbol, int):
            with self._lock:
                resolved_symbol = self._symbol_by_token.get(symbol)
            if not resolved_symbol:
                return None
        else:
            resolved_symbol = enforce_canonical(normalize_symbol(symbol)) or symbol

        with self._lock:
            tick = self._tick_cache.get(resolved_symbol)
            return dict(tick) if tick is not None else None

    async def ensure_fresh_tick(self, symbol: str) -> dict[str, Any] | None:
        """Args: symbol; Returns: cached tick; Raises: none."""

        normalized_symbol = enforce_canonical(normalize_symbol(symbol)) or symbol
        tick = self.get_latest_tick(normalized_symbol)
        stale_threshold = max(float(self._tick_stale_threshold_ms) / 1000.0, 0.0)
        tick_age = self.time_since_last_tick(normalized_symbol)
        tick_stale = tick is None or (
            tick_age is not None
            and stale_threshold > 0.0
            and tick_age > stale_threshold
        )
        ws_disconnected = not self._is_ws_connected()

        if tick_stale and ws_disconnected:
            self._schedule_rest_refresh(normalized_symbol)
        return tick

    def time_since_last_tick(self, symbol: str) -> float | None:
        """Args: symbol; Returns: seconds since last tick; Raises: none."""

        normalized_symbol = enforce_canonical(normalize_symbol(symbol)) or symbol
        with self._lock:
            wallclock = self._last_tick_time.get(normalized_symbol)
            if wallclock is None:
                tick = self._tick_cache.get(normalized_symbol)
                if tick is None:
                    return None
                wallclock = float(tick.get("timestamp") or 0.0)
        if not wallclock:
            return None
        return max(time.time() - float(wallclock), 0.0)

    async def _rest_refresh(self, symbol: str) -> None:
        """Args: symbol; Returns: none; Raises: none."""

        normalized_symbol = enforce_canonical(normalize_symbol(symbol)) or symbol
        with self._lock:
            if normalized_symbol in self._rest_refresh_inflight:
                return
            self._rest_refresh_inflight.add(normalized_symbol)
        try:
            await asyncio.to_thread(self.pull_quote, normalized_symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _rest_refresh: %s",
                exc,
                extra={"event": "mdm_rest_refresh_error", "symbol": normalized_symbol},
                exc_info=exc,
            )
        finally:
            with self._lock:
                self._rest_refresh_inflight.discard(normalized_symbol)

    def _schedule_rest_refresh(self, symbol: str) -> None:
        """Args: symbol; Returns: none; Raises: none."""

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._rest_refresh(symbol))
        except RuntimeError:
            thread = threading.Thread(
                target=lambda: asyncio.run(self._rest_refresh(symbol)),
                name=f"mdm-rest-refresh-{symbol}",
                daemon=True,
            )
            thread.start()

    async def wait_for_live_tick(
        self, token: int, timeout: float = 5
    ) -> dict[str, Any]:
        """Args: token, timeout; Returns: fresh tick; Raises: RuntimeError."""
        start = time.time()
        while time.time() - start < timeout:
            tick = self.get_latest_tick(token)
            if tick:
                return tick
            await asyncio.sleep(0.1)
        raise RuntimeError("Live tick unavailable")

    def get_latest_price(self, symbol: str) -> float | None:
        tick = self.get_latest_tick(symbol)
        if tick is not None:
            try:
                return float(tick["ltp"])
            except (KeyError, TypeError, ValueError):
                pass

            # 🔴 REST fallback (already available in this class)
            broker = getattr(self, "_broker", None)
            if broker:
                try:
                    quote = broker.get_quote(symbol)
                    if isinstance(quote, dict):
                        price = quote.get("last_price") or quote.get("ltp")
                        if price:
                            return float(price)
                except Exception:
                    pass

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
        """Fetch a broker quote for *symbol* and update caches.

        Args:
            symbol: Trading symbol identifier used for the request.

        Returns:
            dict[str, Any]: Normalized quote when available; otherwise a
            best-effort payload containing the raw broker response.

        Raises:
            None.
        """

        candidates: list[str | int] = self._candidate_quote_keys(symbol)
        if not candidates:
            candidates = [symbol]
        quote: dict[str, Any] | None = None
        for key in candidates:
            broker_key: str | int
            if isinstance(key, int):
                broker_key = int(key)
            else:
                broker_key = key
            fetched = self._broker_quote_any(broker_key)
            if fetched:
                quote = fetched
                break
        if quote is None:
            try:
                raw_quote = self._broker.get_quote(symbol)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in pull_quote direct get_quote: %s",
                    exc,
                    extra={
                        "event": "mdm_pull_quote_direct_error",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                    exc_info=exc,
                )
                raw_quote = None
            if isinstance(raw_quote, Mapping):
                quote = dict(raw_quote)
        if quote is None:
            return {"symbol": symbol}
        with self._lock:
            previous = self._latest_ticks.get(symbol)

        normalized = self._normalize_tick(symbol, quote, previous)
        if normalized is not None:
            if not self._is_duplicate(symbol, normalized):
                self._emit_tick(symbol, normalized, source="rest")
            else:
                self._store_tick(symbol, normalized)
            return normalized
        return {"symbol": symbol, **quote}

    def probe_quote(self, symbol: str) -> dict[str, Any]:
        """Run a deep quote diagnostic across resolver, broker, and cache.

        Args:
            symbol: Symbol or token string to inspect.

        Returns:
            dict[str, Any]: Structured diagnostic snapshot.

        Raises:
            None.
        """

        normalized_symbol = (symbol or "").strip().upper()
        self._logger.debug(
            "Entered probe_quote",
            extra={
                "event": "mdm_probe_quote_enter",
                "symbol": normalized_symbol,
            },
        )
        report: dict[str, Any] = {
            "symbol": normalized_symbol,
            "resolver": {},
            "cache": {},
            "broker_raw": {},
            "normalized": {},
            "transport": {},
            "hints": [],
        }
        if not normalized_symbol:
            self._logger.info(
                "Condition met: mdm_probe_quote_blank_symbol",
                extra={"event": "mdm_probe_quote_blank_symbol"},
            )
            report["hints"] = ["Symbol is empty; provide a valid option or token."]
            return report

        token: int | str | None = None
        exchange: str | None = None
        segment = self._margin_segment
        try:
            token = self._token_by_symbol.get(normalized_symbol)
        except Exception as token_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in probe_quote token lookup: %s",
                token_exc,
                extra={"event": "mdm_probe_quote_token_error"},
                exc_info=token_exc,
            )
            token = None
        if token is None and self._resolver is not None:
            try:
                token = self._resolver.resolve(normalized_symbol)
            except Exception as resolve_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote resolver: %s",
                    resolve_exc,
                    extra={"event": "mdm_probe_quote_resolver_error"},
                    exc_info=resolve_exc,
                )
                token = None
        if self._resolver is not None:
            try:
                exchange = getattr(
                    self._resolver, "exchange_for_symbol", lambda _: None
                )(normalized_symbol)
            except Exception as exchange_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote exchange lookup: %s",
                    exchange_exc,
                    extra={"event": "mdm_probe_quote_exchange_error"},
                    exc_info=exchange_exc,
                )
                exchange = None
        report["resolver"] = {
            "ok": bool(token) or bool(exchange),
            "token": token,
            "exchange": exchange,
            "segment": segment,
        }

        candidate_keys = self._candidate_quote_keys(normalized_symbol)

        latest: Mapping[str, Any] | None = None
        try:
            latest = self.get_latest_tick(normalized_symbol)
        except Exception as cache_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in probe_quote cache fetch: %s",
                cache_exc,
                extra={"event": "mdm_probe_quote_cache_error"},
                exc_info=cache_exc,
            )
            latest = None
        ltp = latest.get("ltp") if isinstance(latest, Mapping) else None
        bid = latest.get("bid") if isinstance(latest, Mapping) else None
        ask = latest.get("ask") if isinstance(latest, Mapping) else None
        ts_value = latest.get("timestamp") if isinstance(latest, Mapping) else None
        age_seconds: float | None = None
        if isinstance(ts_value, (int, float)) and ts_value > 0:
            try:
                age_seconds = max(
                    0.0, datetime.now(timezone.utc).timestamp() - float(ts_value)
                )
            except Exception as age_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote cache age: %s",
                    age_exc,
                    extra={"event": "mdm_probe_quote_age_error"},
                    exc_info=age_exc,
                )
                age_seconds = None
        try:
            source = self._last_tick_source.get(normalized_symbol)
        except Exception as source_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in probe_quote cache source: %s",
                source_exc,
                extra={"event": "mdm_probe_quote_source_error"},
                exc_info=source_exc,
            )
            source = None
        report["cache"] = {
            "has_tick": latest is not None,
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "ts": ts_value,
            "age_s": age_seconds,
            "source": source,
        }

        raw_dict: dict[str, Any] | None = None
        broker_ok = False
        try:
            broker = getattr(self, "_broker", None)
            if broker is not None:
                candidate_keys = self._candidate_quote_keys(normalized_symbol)
                lookup: list[str | int] = []
                numeric_token: int | None = None
                if token is not None:
                    try:
                        numeric_token = int(token)
                    except (TypeError, ValueError):
                        self._logger.info(
                            "Condition met: mdm_probe_quote_invalid_token",
                            extra={
                                "event": "mdm_probe_quote_invalid_token",
                                "symbol": normalized_symbol,
                                "token": token,
                            },
                        )
                    else:
                        lookup.append(numeric_token)
                lookup.extend(candidate_keys)
                if lookup:
                    lookup = list(dict.fromkeys(lookup))
                else:
                    lookup = [normalized_symbol]
                quote_any_fn = getattr(broker, "quote_any", None)
                if callable(quote_any_fn) and lookup:
                    try:
                        self._logger.info(
                            "Condition met: mdm_probe_quote_quote_any",
                            extra={
                                "event": "mdm_probe_quote_quote_any",
                                "count": len(lookup),
                            },
                        )
                        raw_any = quote_any_fn(lookup)  # type: ignore[arg-type]
                    except Exception as quote_any_exc:  # noqa: BLE001
                        self._logger.error(
                            "Failure in probe_quote quote_any: %s",
                            quote_any_exc,
                            extra={"event": "mdm_probe_quote_quote_any_error"},
                            exc_info=quote_any_exc,
                        )
                    else:
                        if isinstance(raw_any, Mapping) and raw_any:
                            raw_dict = dict(raw_any)
                            broker_ok = True
                if not broker_ok:
                    for key in lookup:
                        broker_key: str | int
                        if isinstance(key, int):
                            broker_key = int(key)
                        else:
                            broker_key = key
                        candidate = self._broker_quote_any(broker_key)
                        if candidate:
                            raw_dict = dict(candidate)
                            broker_ok = True
                            break
            else:
                self._logger.info(
                    "Condition met: mdm_probe_quote_no_broker",
                    extra={"event": "mdm_probe_quote_no_broker"},
                )
        except Exception as broker_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in probe_quote broker fetch: %s",
                broker_exc,
                extra={"event": "mdm_probe_quote_broker_error"},
                exc_info=broker_exc,
            )
            broker_ok = False
        raw_keys: list[str] = []
        if raw_dict is not None:
            try:
                raw_keys = [str(key) for key in list(raw_dict.keys())[:10]]
            except Exception as key_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote raw key inspection: %s",
                    key_exc,
                    extra={"event": "mdm_probe_quote_rawkey_error"},
                    exc_info=key_exc,
                )
                raw_keys = []
        sample = None
        if raw_dict is not None and raw_keys:
            try:
                sample = {key: raw_dict[key] for key in raw_keys if key in raw_dict}
            except Exception as sample_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote sample build: %s",
                    sample_exc,
                    extra={"event": "mdm_probe_quote_sample_error"},
                    exc_info=sample_exc,
                )
                sample = None
        report["broker_raw"] = {
            "ok": broker_ok,
            "keys": raw_keys,
            "sample": sample,
        }

        normalized: Mapping[str, Any] | None = None
        norm_ok = False
        normalization_source: dict[str, Any] | None = None
        if broker_ok and raw_dict is not None:
            payload_candidates: list[str] = []
            if normalized_symbol:
                payload_candidates.append(normalized_symbol)
            for key in candidate_keys:
                if isinstance(key, str) and key not in payload_candidates:
                    payload_candidates.append(key)
            if token is not None:
                token_alias = str(token)
                if token_alias not in payload_candidates:
                    payload_candidates.append(token_alias)
                try:
                    canonical_token = str(int(token))
                except (TypeError, ValueError):
                    canonical_token = None
                else:
                    if (
                        canonical_token is not None
                        and canonical_token not in payload_candidates
                    ):
                        payload_candidates.append(canonical_token)
            for alias in payload_candidates:
                candidate_payload = raw_dict.get(alias)
                if isinstance(candidate_payload, Mapping):
                    normalization_source = dict(candidate_payload)
                    break
            if normalization_source is None:
                for value in raw_dict.values():
                    if isinstance(value, Mapping):
                        normalization_source = dict(value)
                        break
            if normalization_source is None:
                normalization_source = dict(raw_dict)
            previous_tick: Mapping[str, Any] | None
            with self._lock:
                previous_tick = self._latest_ticks.get(normalized_symbol)
            try:
                normalized = self._normalize_tick(
                    normalized_symbol,
                    normalization_source,
                    previous_tick,
                )
            except Exception as norm_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in probe_quote normalization: %s",
                    norm_exc,
                    extra={"event": "mdm_probe_quote_normalize_error"},
                    exc_info=norm_exc,
                )
                normalized = None
            if isinstance(normalized, Mapping):
                try:
                    norm_ok = float(normalized.get("ltp", 0.0) or 0.0) > 0.0
                except Exception as norm_check_exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in probe_quote normalization check: %s",
                        norm_check_exc,
                        extra={"event": "mdm_probe_quote_normcheck_error"},
                        exc_info=norm_check_exc,
                    )
                    norm_ok = False
        report["normalized"] = {
            "ok": norm_ok,
            "ltp": normalized.get("ltp") if isinstance(normalized, Mapping) else None,
            "bid": normalized.get("bid") if isinstance(normalized, Mapping) else None,
            "ask": normalized.get("ask") if isinstance(normalized, Mapping) else None,
            "ts": (
                normalized.get("timestamp") if isinstance(normalized, Mapping) else None
            ),
            # additional diagnostics
            "has_depth": bool(
                (isinstance(normalized, Mapping) and normalized.get("depth"))
                or (isinstance(latest, Mapping) and latest.get("depth"))
            ),
            "source": (
                normalized.get("_source") if isinstance(normalized, Mapping) else None
            ),
        }

        # earlier report["cache"] is already built — ensure it includes these:
        report["cache"].update(
            {
                "has_tick": latest is not None,
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "ts": ts_value,
                "age_s": age_seconds,
                "source": source,
            }
        )
        ws_enabled = self._ws is not None
        ws_connected = bool(self._ws_connected)
        poll_enabled = bool(self._rest_poll_enabled)
        report["transport"] = {
            "ws_enabled": ws_enabled,
            "ws_connected": ws_connected,
            "poll_enabled": poll_enabled,
            "poll_interval": float(self._rest_poll_interval),
            "poll_max": int(self._rest_poll_max_symbols),
        }

        hints: list[str] = []
        if not report["resolver"]["ok"]:
            hints.append("Resolver missing token/exchange; verify symbol mapping.")
        if not broker_ok:
            hints.append("Broker get_quote returned empty; validate symbol and auth.")
        if broker_ok and not norm_ok:
            hints.append("Normalization failed; extend _normalize_tick for payload.")
        if not report["cache"]["has_tick"]:
            hints.append("Cache empty; ensure polling or websocket subscriptions.")
        if not ws_enabled and not poll_enabled:
            hints.append("No transport enabled; enable websocket or REST polling.")
        report["hints"] = hints

        self._logger.info(
            "Condition met: mdm_probe_quote_complete",
            extra={
                "event": "mdm_probe_quote_complete",
                "symbol": normalized_symbol,
                "hints": len(hints),
            },
        )
        return report

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
            self._logger.error(
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

        self._logger.debug(
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
        self._logger.error(
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

    def update_hydration_status(
        self, symbol: str, bars: Sequence[Mapping[str, Any]]
    ) -> None:
        """Update hydration status from bars. Args: symbol, bars. Returns: None. Raises: None."""
        try:
            normalized = normalize_symbol(str(symbol or ""))
            bar_count = len(list(bars))
            if bar_count >= 20:
                self._hydration_status[normalized] = "READY"
            else:
                self._logger.error(
                    "insufficient_bars_for_strategy",
                    extra={
                        "event": "insufficient_bars_for_strategy",
                        "symbol": normalized,
                        "bars": bar_count,
                    },
                )
        except Exception as exc:
            self._logger.error(
                "Failure in update_hydration_status: %s", exc, exc_info=exc
            )

    def get_hydration_status(self, symbol: str) -> str:
        """Return hydration status. Args: symbol. Returns: status string. Raises: None."""
        normalized = normalize_symbol(str(symbol or ""))
        return self._hydration_status.get(normalized, "HYDRATING")

    @property
    def ws_connected(self) -> bool:
        """Expose WebSocket connectivity for compatibility."""

        return self._ws_connected

    def transport_status(self) -> dict[str, Any]:
        """Return websocket and REST polling state snapshot for diagnostics.

        Args:
            None.

        Returns:
            dict[str, Any]: Snapshot including websocket connectivity and
            polling configuration.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketDataManager.transport_status",
            extra={"event": "mdm_transport_status_enter"},
        )
        status: dict[str, Any] = {
            "ws_connected": False,
            "poll_enabled": False,
            "poll_interval": 0.0,
            "poll_max": 0,
        }
        try:
            ws_connected = bool(self._ws_connected)
            if self._ws is not None:
                try:
                    ws_connected = bool(self._ws.is_connected())
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self._logger.debug(
                        "Failure in MarketDataManager.transport_status ws probe: %s",
                        exc,
                        extra={"event": "mdm_transport_status_ws_probe_error"},
                    )
            poll_enabled = bool(self._rest_poll_enabled)
            poll_interval = float(self._rest_poll_interval or 0.0)
            poll_max = int(self._rest_poll_max_symbols or 0)
            status.update(
                {
                    "ws_connected": ws_connected,
                    "poll_enabled": poll_enabled,
                    "poll_interval": poll_interval,
                    "poll_max": poll_max,
                }
            )
            self._logger.info(
                "Condition met: mdm_transport_status_ready",
                extra={
                    "event": "mdm_transport_status_ready",
                    "ws_connected": ws_connected,
                    "poll_enabled": poll_enabled,
                    "poll_interval": poll_interval,
                    "poll_max": poll_max,
                },
            )
            return status
        except Exception as exc:  # noqa: BLE001 - defensive safeguard
            self._logger.error(
                "Failure in MarketDataManager.transport_status: %s",
                exc,
                extra={"event": "mdm_transport_status_error"},
                exc_info=exc,
            )
            return status

    def resolve_symbol_token(self, symbol: str) -> int | None:
        """Best-effort token resolver combining resolver and cache fallbacks.

        Args:
            symbol: Trading symbol to resolve into an instrument token.

        Returns:
            int | None: Resolved instrument token when available, otherwise
            ``None``.

        Raises:
            None.
        """

        normalized = (symbol or "").strip()
        self._logger.debug(
            "Entered MarketDataManager.resolve_symbol_token",
            extra={
                "event": "mdm_resolve_symbol_token_enter",
                "symbol": normalized or symbol,
            },
        )
        if not normalized:
            self._logger.info(
                "Condition met: mdm_resolve_symbol_token_empty",
                extra={"event": "mdm_resolve_symbol_token_empty"},
            )
            return None
        token_source = "none"
        try:
            resolver = getattr(self, "_resolver", None)
            if resolver is not None and callable(getattr(resolver, "resolve", None)):
                try:
                    resolved = resolver.resolve(normalized)
                    if resolved:
                        resolved_token_int = int(resolved)
                        if resolved_token_int > 0:
                            token_source = "instrument_resolver"
                            self._logger.info(
                                "Condition met: mdm_resolve_symbol_token_ready",
                                extra={
                                    "event": "mdm_resolve_symbol_token_ready",
                                    "symbol": normalized,
                                    "token": resolved_token_int,
                                    "source": token_source,
                                },
                            )
                            return resolved_token_int
                except Exception as exc:  # noqa: BLE001 - resolver failure
                    self._logger.error(
                        "MDM.resolve_symbol_token resolver failed: %s",
                        exc,
                        extra={
                            "event": "mdm_resolve_symbol_token_resolver_error",
                            "symbol": normalized,
                        },
                        exc_info=exc,
                    )

            try:
                resolved_token = self._resolve_token(normalized)
                if resolved_token:
                    resolved_token_int = int(resolved_token)
                    token_source = "mdm_resolver"
                    self._logger.info(
                        "Condition met: mdm_resolve_symbol_token_ready",
                        extra={
                            "event": "mdm_resolve_symbol_token_ready",
                            "symbol": normalized,
                            "token": resolved_token_int,
                            "source": token_source,
                        },
                    )
                    return resolved_token_int
            except Exception as exc:  # noqa: BLE001 - defensive guard
                self._logger.error(
                    "Failure in MarketDataManager.resolve_symbol_token internal: %s",
                    exc,
                    extra={
                        "event": "mdm_resolve_symbol_token_internal_error",
                        "symbol": normalized,
                    },
                    exc_info=exc,
                )

            try:
                token_map = getattr(self, "_token_by_symbol", {}) or {}
                cached = token_map.get(normalized)
                if cached is None:
                    cached = token_map.get(normalized.upper())
                if cached:
                    cached_token_int = int(cached)
                    if cached_token_int > 0:
                        token_source = "cache"
                        self._logger.info(
                            "Condition met: mdm_resolve_symbol_token_ready",
                            extra={
                                "event": "mdm_resolve_symbol_token_ready",
                                "symbol": normalized,
                                "token": cached_token_int,
                                "source": token_source,
                            },
                        )
                        return cached_token_int
            except Exception as exc:  # noqa: BLE001 - cache fallback
                self._logger.error(
                    "Failure in MarketDataManager.resolve_symbol_token cache: %s",
                    exc,
                    extra={
                        "event": "mdm_resolve_symbol_token_cache_error",
                        "symbol": normalized,
                    },
                    exc_info=exc,
                )

            self._logger.info(
                "Condition met: mdm_resolve_symbol_token_miss",
                extra={
                    "event": "mdm_resolve_symbol_token_miss",
                    "symbol": normalized,
                },
            )
            return None
        except Exception as exc:  # noqa: BLE001 - defensive safeguard
            self._logger.error(
                "Failure in MarketDataManager.resolve_symbol_token: %s",
                exc,
                extra={
                    "event": "mdm_resolve_symbol_token_error",
                    "symbol": normalized,
                },
                exc_info=exc,
            )
            return None

    def is_live(self) -> bool:
        if self._ws is None:
            return True
        state = self._ws.connection_state()
        if state != ConnectionState.CONNECTED:
            return self._rest_poll_enabled and self._has_recent_rest_ticks()
        if not self._ws.is_connected():
            return self._rest_poll_enabled and self._has_recent_rest_ticks()
        return True

    def attach_tick_bus(self, tick_bus: Any) -> None:
        """Args: tick_bus; Returns: none; Raises: none."""
        try:
            # TickBus is used as outbound fan-out only (MDM -> DataHub).
            # Do not subscribe MDM back onto TickBus to keep a single tick path.
            self._tick_bus = tick_bus
        except Exception as e:
            self._logger.error("Failure in MarketDataManager.attach_tick_bus: %s", e)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Args: loop; Returns: none; Raises: none."""
        try:
            if self._main_loop is not None:
                self._logger.warning("Event loop already wired — ignoring rewire")
                return
            self._main_loop = loop
        except Exception as e:
            self._logger.error("Failure in MarketDataManager.set_event_loop: %s", e)

    def _on_tick(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol_value = tick.get("symbol")
            if not symbol_value:
                return
            symbol = enforce_canonical(normalize_symbol(str(symbol_value)))
            incoming = dict(tick)
            incoming_ts = float(incoming.get("timestamp") or time.time())
            previous = self._tick_cache.get(symbol)
            if previous is not None:
                previous_ts = float(previous.get("timestamp") or 0.0)
                previous_source = str(previous.get("source", "")).lower()
                incoming_source = str(incoming.get("source", "")).lower()
                if (
                    previous_source == "ws"
                    and incoming_source in {"rest", "polling"}
                    and incoming_ts <= previous_ts
                ):
                    return
            self._tick_cache[symbol] = incoming
            self._last_tick_time[symbol] = time.time()
            self._handle_tick(incoming)
        except Exception as e:
            self._logger.error("Failure in MarketDataManager._on_tick: %s", e)

    # ------------------------------------------------------------------
    # Internal plumbing

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        """Process an incoming raw tick from WebSocket or Polling."""

        if not isinstance(tick, dict):
            self._logger.error("Invalid tick format: %s", type(tick))
            return

        instrument_token = tick.get("instrument_token")
        last_price = tick.get("last_price")
        if instrument_token is None or last_price is None:
            return

        self._last_tick_time["__global__"] = time.monotonic()

        token: int | None = None
        try:
            token = int(instrument_token)
        except (ValueError, TypeError):
            token = None

        symbol = None
        if token is not None:
            resolver = getattr(self, "_resolver", None)
            resolve_token = getattr(resolver, "resolve_token", None)
            if callable(resolve_token):
                symbol = resolve_token(token)
            if not symbol:
                symbol = self._symbol_by_token.get(token)

        if not symbol:
            self._logger.error(
                "Token %s not mapped to symbol — dropping tick",
                instrument_token,
            )
            return

        symbol = enforce_canonical(normalize_symbol(str(symbol)))
        if symbol.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {symbol}")

        if symbol not in self._tracked_symbols:
            self._tracked_symbols.add(symbol)

        with self._lock:
            previous = self._latest_ticks.get(symbol)

        try:
            normalized_tick = self._normalize_tick(symbol, tick, previous)
        except Exception as exc:
            self._logger.error(
                f"mdm_normalize_crash: {exc}", extra={"symbol": symbol}, exc_info=True
            )
            return

        if not normalized_tick:
            return

        if self._is_duplicate(symbol, normalized_tick):
            return

        if self._ws:
            self.set_ws_connected(True)
        self.bump_heartbeat()
        log_throttled(
            self._logger,
            f"live_tick_{symbol}",
            f"LIVE_TICK {symbol} {last_price}",
            interval_sec=5.0,
            level=logging.DEBUG,
        )
        self._tick_stats[symbol] += 1
        now = time.monotonic()
        if now - self._last_tick_stats_log >= 15.0:
            summary = ", ".join(
                f"{sym}:{cnt}" for sym, cnt in sorted(self._tick_stats.items())
            )
            self._logger.debug(f"TICK_RATE_15S {summary}")
            self._tick_stats.clear()
            self._last_tick_stats_log = now
        self._emit_tick(symbol, normalized_tick, source="ws")

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

    def _store_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        """Persist normalized *tick* for *symbol* and refresh derived series."""

        wallclock = tick.get("timestamp", time.time())

        # 🔥 PRODUCTION FIX — enforce canonical key
        symbol = enforce_canonical(normalize_symbol(symbol))

        cached_tick = dict(tick)
        with self._lock:
            self._latest_ticks[symbol] = cached_tick
            self._tick_cache[symbol] = cached_tick
            self._last_tick_time[symbol] = time.time()
            self._history[symbol].append(cached_tick)
            self._last_tick_wallclock[symbol] = float(wallclock)
        self._notify_unified_manager(symbol, cached_tick)
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
        source = str(source or "unknown").lower()
        self._store_tick(symbol, tick)
        callbacks: list[TickCallback]
        tick_payload = dict(tick)
        with self._lock:
            self._last_tick_source[symbol] = source
            callbacks = list(self._subscribers.get(symbol, ()))
            self._tick_counter += 1
        if source != "ws":
            self.bump_heartbeat()
        now_mono = time.monotonic()
        tick_stats_interval = float(os.getenv("TICK_STATS_INTERVAL", "5.0"))
        if now_mono - self._last_tick_log_time >= tick_stats_interval:
            self._logger.debug(
                "EVENT|tick_stats|cached=%d|ticks_last_5s=%d",
                len(self._tick_cache),
                self._tick_counter,
            )
            self._tick_counter = 0
            self._last_tick_log_time = now_mono
        try:
            self._m_ticks.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    loop = self._main_loop
                    if loop is not None and loop.is_running():
                        loop.call_soon_threadsafe(
                            lambda: loop.create_task(callback(dict(tick_payload)))
                        )
                    else:
                        with self._lock:
                            self._async_dispatch_drops += 1
                            async_dispatch_drops = self._async_dispatch_drops
                        now = time.monotonic()
                        if now - self._last_async_drop_log > 5.0:
                            self._logger.warning(
                                "Async tick dispatch drops=%d",
                                async_dispatch_drops,
                                extra={"event": "async_dispatch_drops"},
                            )
                            self._last_async_drop_log = now
                        self._logger.warning(
                            "Async tick callback dropped — main loop not wired",
                            extra={
                                "event": "async_callback_no_loop",
                                "symbol": symbol,
                            },
                        )
                else:
                    callback(dict(tick_payload))
            except Exception as exc:
                self._logger.error(
                    "Tick callback failed", extra={"symbol": symbol, "error": str(exc)}
                )

    def _is_ws_connected(self) -> bool:
        """Args: none; Returns: websocket connectivity; Raises: none."""

        try:
            if self._ws is None:
                return bool(self._ws_connected)
            return bool(self._ws.is_connected())
        except Exception:  # noqa: BLE001
            return bool(self._ws_connected)

    def _start_health_monitor(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        if (
            self._health_monitor_thread is not None
            and self._health_monitor_thread.is_alive()
        ):
            return
        self._health_monitor_stop.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="mdm-health-monitor",
            daemon=True,
        )
        self._health_monitor_thread.start()

    def _health_monitor_loop(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        while not self._health_monitor_stop.wait(1.0):
            try:
                self._check_zombie_ticks()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _health_monitor_loop: %s",
                    exc,
                    extra={"event": "mdm_health_monitor_error"},
                    exc_info=exc,
                )

    def _check_zombie_ticks(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        from nifty_scalper_bot.utils.market_hours import is_market_open

        if not is_market_open():
            self._zombie_stale_logged = False
            return
        if not self._is_ws_connected():
            self._zombie_stale_logged = False
            return

        tick_age = self.time_since_last_tick(self._zombie_symbol)
        if tick_age is None or tick_age <= self._zombie_tick_threshold_sec:
            self._zombie_stale_logged = False
            return

        if not self._zombie_stale_logged:
            self._logger.critical(
                "CRITICAL zombie_tick_detected symbol=%s age=%.2fs threshold=%.2fs",
                self._zombie_symbol,
                tick_age,
                self._zombie_tick_threshold_sec,
                extra={
                    "event": "mdm_zombie_tick_detected",
                    "symbol": self._zombie_symbol,
                    "age_seconds": tick_age,
                },
            )
            self._zombie_stale_logged = True

        self._trigger_zombie_ws_restart()

    def _trigger_zombie_ws_restart(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        now = time.monotonic()
        if now < self._zombie_breaker_open_until:
            return

        since_last = now - self._zombie_last_restart_attempt_at
        if since_last < self._zombie_restart_cooldown_sec:
            return
        self._zombie_last_restart_attempt_at = now

        ws = self._ws
        if ws is None:
            return

        reconnect = getattr(ws, "force_reconnect", None)
        if not callable(reconnect):
            self._zombie_restart_failures += 1
        else:
            try:
                reconnect()
                self._zombie_restart_failures = 0
                self._logger.warning(
                    "Condition met: mdm_zombie_ws_restart",
                    extra={
                        "event": "mdm_zombie_ws_restart",
                        "failure_count": self._zombie_restart_failures,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._zombie_restart_failures += 1
                self._logger.error(
                    "Failure in _trigger_zombie_ws_restart: %s",
                    exc,
                    extra={"event": "mdm_zombie_ws_restart_error"},
                    exc_info=exc,
                )

        if self._zombie_restart_failures > self._zombie_restart_limit:
            self._zombie_breaker_open_until = now + self._zombie_restart_window
            self._zombie_restart_failures = 0
            self._logger.error(
                "Failure in _trigger_zombie_ws_restart: circuit breaker open",
                extra={
                    "event": "mdm_zombie_circuit_open",
                    "open_seconds": self._zombie_restart_window,
                },
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
        """High-frequency batched polling loop (Production-Grade).

        Optimized for <500ms latency. Batches requests and uses smart sleep.
        Includes self-healing logic for session expiry and rate limits.
        """
        # SCOUT CONFIGURATION: 0.5s target interval
        target_interval = 1.5
        self._logger.info(
            f"🚀 Scout Polling Started. Target Interval: {target_interval}s",
            extra={"event": "scout_poll_started"},
        )

        # Move margin checks to a slower cadence (60s) to prevent blocking ticks
        last_margin_refresh = 0.0
        margin_interval = 60.0
        consecutive_errors = 0

        while not self._rest_poll_stop.is_set():
            loop_start = time.time()

            try:
                # 1. Throttled Margin Refresh (Non-Blocking Priority)
                if (loop_start - last_margin_refresh) > margin_interval:
                    try:
                        self.refresh_margin_snapshot()
                        last_margin_refresh = loop_start
                    except Exception as exc:
                        self._logger.error(f"Margin refresh failed: {exc}")

                # 2. Identify Symbols
                symbols = self._symbols_for_poll()
                if not symbols:
                    # Idle wait if nothing to track
                    if self._rest_poll_stop.wait(1.0):
                        break
                    continue

                # 3. Batch Fetch (Critical Optimization)
                # Zerodha accepts up to 500 symbols per call. using 200 for safety.
                batch_size = 200
                for i in range(0, len(symbols), batch_size):
                    if self._rest_poll_stop.is_set():
                        break

                    batch = symbols[i : i + batch_size]
                    try:
                        # Fetch all quotes in ONE HTTP call
                        quotes = self._broker.quote_any(batch) or {}
                        arrival_time = time.time()

                        # Ingest immediately
                        for symbol, data in quotes.items():
                            # CRITICAL: Inject local timestamp for Stale Data logic
                            data["_local_timestamp"] = arrival_time
                            self.ingest_rest_quote(symbol, data)

                        # Success - Reset error counters
                        consecutive_errors = 0

                    except Exception as exc:
                        # [FIX] Handle Batch-Level Failures
                        self._logger.warning(
                            f"Batch poll failed for {len(batch)} symbols: {exc}",
                            extra={"event": "scout_batch_fail"},
                        )
                        raise  # Re-raise to trigger main loop error handling logic below

            except Exception as exc:
                error_msg = str(exc).lower()
                consecutive_errors += 1

                # [FIX 1] DETECT FATAL SESSION ERRORS (Cure for Zombie Mode)
                # If broker says "Forbidden", "Unauthorized", or "Access Denied", our session is dead.
                # We must kill the process so Docker/Railway restarts it with a fresh session.
                if (
                    "403" in error_msg
                    or "401" in error_msg
                    or "unauthorized" in error_msg
                    or "access denied" in error_msg
                ):
                    self._logger.critical(
                        "🚨 FATAL: Broker Session Expired. Killing process to force auto-restart.",
                        extra={"event": "scout_session_expired", "error": str(exc)},
                    )
                    import os

                    os._exit(1)  # Hard exit to ensure restart

                # [FIX 2] Handle Rate Limits Gracefully
                elif "rate limit" in error_msg or "429" in error_msg:
                    self._logger.warning(
                        "⚠️ Rate Limit Hit! Cooling down Scout Poller...",
                        extra={"event": "scout_rate_limit"},
                    )
                    time.sleep(5.0)  # Hard wait
                    target_interval = min(
                        5.0, target_interval + 0.5
                    )  # Permanently slow down

                # [FIX 3] Exponential Backoff for Network Blips
                else:
                    backoff = min(30.0, 0.5 * (2**consecutive_errors))
                    self._logger.error(
                        f"Scout Loop Error (Attempt {consecutive_errors}): {exc}",
                        exc_info=True,
                        extra={"event": "scout_critical_error", "backoff": backoff},
                    )
                    time.sleep(backoff)

            # 4. Smart Sleep (Maintain Rhythm)
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, target_interval - elapsed)

            if self._rest_poll_stop.wait(sleep_time):
                break

    def _symbols_for_poll(self) -> list[str]:
        """Derive poll candidates across subscribers, cache, and tracking.

        Args:
            None.

        Returns:
            Ordered list of symbols selected for REST polling.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _symbols_for_poll",
            extra={"event": "mdm_symbols_for_poll_enter"},
        )
        try:
            with self._lock:
                candidates = set(self._subscribers.keys())
                if not candidates:
                    candidates.update(self._latest_ticks.keys())
                candidates.update(self._tracked_symbols)
            ordered = sorted(symbol for symbol in candidates if symbol)
            limit = self._rest_poll_max_symbols
            if limit > 0 and len(ordered) > limit:
                self._logger.debug(
                    "Condition met: mdm_symbols_for_poll_limited",
                    extra={
                        "event": "mdm_symbols_for_poll_limited",
                        "limit": limit,
                        "count": len(ordered),
                    },
                )
                ordered = ordered[:limit]
            ceiling = self._poll_batch_ceiling
            if ceiling > 0 and len(ordered) > ceiling:
                ordered = ordered[:ceiling]
            return ordered
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _symbols_for_poll: %s",
                exc,
                extra={"event": "mdm_symbols_for_poll_error"},
                exc_info=exc,
            )
            return []
        # diagnostic batch log
        try:
            self._logger.info(
                "REST polling batch",
                extra={
                    "event": "mdm_rest_poll_batch",
                    "batch_size": len(symbols),
                    "symbols_preview": symbols[:5],
                    "poll_interval": float(self._rest_poll_interval),
                },
            )
        except Exception:
            # don't let diagnostics break the loop
            pass

    def ensure_tracking(self, symbol: str, *, seed: bool = True) -> bool:
        """Ensure *symbol* is tracked for REST polling and optional seeding.

        Args:
            symbol: Trading symbol identifier to watch.
            seed: Flag indicating whether to seed the cache from broker REST.

        Returns:
            ``True`` when the symbol was accepted for tracking, else ``False``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered ensure_tracking",
            extra={"event": "mdm_ensure_tracking_enter", "symbol": symbol},
        )
        sym = enforce_canonical(normalize_symbol(str(symbol or "")))
        if not sym:
            self._logger.info(
                "Condition met: mdm_ensure_tracking_blank",
                extra={"event": "mdm_ensure_tracking_blank"},
            )
            return False
        if sym.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {sym}")
        try:
            with self._lock:
                self._tracked_symbols.add(sym)
            self._logger.info(
                "Condition met: mdm_tracking_added",
                extra={"event": "mdm_tracking_added", "symbol": sym},
            )
            if seed and not self._seed_completed and not self._ws_connected:
                if sym not in self._seeded_symbols:
                    seeded = self._seed_quote_from_broker(sym)
                    if seeded:
                        self._seeded_symbols.add(sym)
                        self._logger.info(
                            "Condition met: mdm_seed_success",
                            extra={"event": "mdm_seed_success", "symbol": sym},
                        )
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in ensure_tracking: %s",
                exc,
                extra={"event": "mdm_ensure_tracking_error", "symbol": sym},
                exc_info=exc,
            )
            return False

    def untrack(self, symbol: str) -> bool:
        """Remove *symbol* from the REST polling tracking set if present.

        Args:
            symbol: Trading symbol identifier to remove.

        Returns:
            ``True`` if the symbol was tracked prior to removal, else ``False``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered untrack",
            extra={"event": "mdm_untrack_enter", "symbol": symbol},
        )
        sym = enforce_canonical(normalize_symbol(str(symbol or "")))
        if not sym:
            self._logger.info(
                "Condition met: mdm_untrack_blank",
                extra={"event": "mdm_untrack_blank"},
            )
            return False
        if sym.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {sym}")
        try:
            with self._lock:
                existed = sym in self._tracked_symbols
                self._tracked_symbols.discard(sym)
            self._logger.info(
                "Condition met: mdm_tracking_removed",
                extra={
                    "event": "mdm_tracking_removed",
                    "symbol": sym,
                    "existed": existed,
                },
            )
            return existed
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in untrack: %s",
                exc,
                extra={"event": "mdm_untrack_error", "symbol": sym},
                exc_info=exc,
            )
            return False

    def list_tracked(self) -> list[str]:
        """Return the sorted list of currently tracked symbols.

        Args:
            None.

        Returns:
            Sorted list of tracked symbol identifiers.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered list_tracked",
            extra={"event": "mdm_list_tracked_enter"},
        )
        try:
            with self._lock:
                snapshot = sorted(self._tracked_symbols)
            return snapshot
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in list_tracked: %s",
                exc,
                extra={"event": "mdm_list_tracked_error"},
                exc_info=exc,
            )
            return []

    def _looks_like_option(self, symbol: str) -> bool:
        """Return whether *symbol* resembles an option or future contract.

        Args:
            symbol: Trading symbol candidate.

        Returns:
            ``True`` when the symbol appears to be a derivative contract.
            ``False`` otherwise.

        Raises:
            None.
        """

        normalized = (symbol or "").upper()
        return "FUT" in normalized or normalized.endswith(("CE", "PE"))

    def is_tracked(self, symbol: str) -> bool:
        """Return ``True`` if *symbol* is currently tracked for polling.

        Args:
            symbol: Trading symbol identifier to inspect.

        Returns:
            Boolean indicating tracking membership.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered is_tracked",
            extra={"event": "mdm_is_tracked_enter", "symbol": symbol},
        )
        try:
            sym = enforce_canonical(normalize_symbol(str(symbol or "")))
            if not sym:
                return False
            with self._lock:
                return sym in self._tracked_symbols
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in is_tracked: %s",
                exc,
                extra={"event": "mdm_is_tracked_error", "symbol": symbol},
                exc_info=exc,
            )
            return False

    def tracked_snapshot(self) -> list[str]:
        """Return a snapshot of tracked symbols for UI layers.

        Args:
            None.

        Returns:
            Sorted list of tracked symbol identifiers.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered tracked_snapshot",
            extra={"event": "mdm_tracked_snapshot_enter"},
        )
        try:
            return self.list_tracked()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in tracked_snapshot: %s",
                exc,
                extra={"event": "mdm_tracked_snapshot_error"},
                exc_info=exc,
            )
            return []

    def _seed_quote_from_broker(self, symbol: str) -> bool:
        """Fetch a single REST quote for *symbol* and seed the cache once."""
        self._logger.debug(
            "Entered _seed_quote_from_broker",
            extra={"event": "mdm_seed_quote_enter", "symbol": symbol},
        )
        outcome = "failure"

        try:
            symbol = enforce_canonical(normalize_symbol(str(symbol)))
            if symbol.count(":") != 1:
                raise RuntimeError(f"Malformed canonical symbol: {symbol}")
            broker = getattr(self, "_broker", None)
            if broker is None or not hasattr(broker, "get_quote"):
                return False
            if self._ws_connected:
                return False
            if symbol in self._seeded_symbols:
                return False

            payload = broker.get_quote(symbol)
            if not isinstance(payload, Mapping) or not payload:
                return False

            ltp = _coerce_positive_float(
                payload.get("last_price")
                or payload.get("ltp")
                or payload.get("LastTradedPrice")
            )
            bid = _coerce_positive_float(payload.get("bid"))
            ask = _coerce_positive_float(payload.get("ask"))

            ts = float(time.time())
            tick = {
                "symbol": symbol,
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "timestamp": ts,
            }

            with self._lock:
                self._latest_ticks[symbol] = tick
                self._last_tick_source[symbol] = "rest"

            self._notify_unified_manager(symbol, tick)
            self._seeded_symbols.add(symbol)
            self._seed_completed = True
            outcome = "success"
            return True

        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _seed_quote_from_broker: %s",
                exc,
                extra={"event": "mdm_seed_quote_error", "symbol": symbol},
                exc_info=exc,
            )
            return False
        finally:
            try:
                METRICS.record_mdm_rest_seed(symbol=symbol, outcome=outcome)
            except Exception as metric_exc:  # noqa: BLE001
                self._logger.debug(
                    "mdm_seed_metric_failed",
                    extra={
                        "event": "mdm_seed_metric_failed",
                        "symbol": symbol,
                        "error": str(metric_exc),
                    },
                )

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

        normalized = (symbol or "").strip().upper()
        if not normalized:
            return []
        base_symbol = normalized.split(":", 1)[-1]
        initial: list[str | int] = []
        if ":" in normalized:
            initial.append(normalized)
            if base_symbol:
                initial.append(base_symbol)
                if self._looks_like_option(base_symbol):
                    initial.append(f"NFO:{base_symbol}")
                else:
                    initial.append(f"NSE:{base_symbol}")
        else:
            if self._looks_like_option(base_symbol):
                initial.append(f"NFO:{base_symbol}")
            else:
                initial.append(f"NSE:{base_symbol}")
            initial.append(base_symbol)
        resolver = getattr(self, "_resolver", None)
        lookup_keys: list[str] = []
        if normalized:
            lookup_keys.append(normalized)
        if base_symbol and base_symbol not in lookup_keys:
            lookup_keys.append(base_symbol)
        for key in lookup_keys:
            if resolver is None:
                break
            try:
                meta = resolver.lookup(key)
            except Exception as exc:  # noqa: BLE001
                self._logger.info(
                    "resolver_lookup_failed",
                    extra={"symbol": key, "error": str(exc)},
                )
                continue
            if not isinstance(meta, Mapping):
                continue
            tradingsymbol = str(meta.get("tradingsymbol") or "").upper()
            exchange = str(meta.get("exchange") or "NFO").upper()
            token = meta.get("instrument_token") or meta.get("token")
            if tradingsymbol:
                initial.append(tradingsymbol)
                initial.append(f"{exchange}:{tradingsymbol}")
            if token is not None:
                try:
                    initial.append(int(token))
                except (TypeError, ValueError):
                    self._logger.info(
                        "resolver_token_invalid",
                        extra={"symbol": key, "token": token},
                    )
        seen: set[str | int] = set()
        ordered: list[str | int] = []
        for candidate in initial:
            if candidate in seen or candidate in {"", None}:
                continue
            seen.add(candidate)
            ordered.append(candidate)

        if not ordered:
            # mapping failure — surface immediately rather than silently fallback
            self._logger.error(
                "Token-to-symbol mapping empty",
                extra={"initial": initial, "lookup_keys": lookup_keys},
                exc_info=False,
            )
            raise ValueError(
                f"Token-to-symbol mapping failed for input: {initial[:10]}"
            )
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
            quote_any_fn = getattr(broker, "quote_any", None)
            if callable(quote_any_fn):
                try:
                    raw_any = quote_any_fn([key])  # type: ignore[arg-type]
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in _broker_quote_any quote_any: %s",
                        exc,
                        extra={"event": "mdm_quote_any_error", "key": key},
                        exc_info=exc,
                    )
                else:
                    if isinstance(raw_any, Mapping) and raw_any:
                        key_candidates: list[str] = []
                        if isinstance(key, int):
                            key_candidates.append(str(int(key)))
                        elif isinstance(key, str):
                            symbol_aliases = [key, key.upper(), key.split(":", 1)[-1]]
                            for alias in symbol_aliases:
                                if alias not in key_candidates:
                                    key_candidates.append(alias)
                        for alias in key_candidates:
                            payload = raw_any.get(alias)
                            if isinstance(payload, Mapping):
                                return dict(payload)
                        for value in raw_any.values():
                            if isinstance(value, Mapping):
                                return dict(value)
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
        candidates: list[str | int] = self._candidate_quote_keys(symbol)
        if not candidates:
            candidates = [symbol]
        self._logger.info(
            "reference_price_refresh",
            extra={
                "symbol": symbol,
                "age_ms": self.quote_age_ms(symbol) if symbol else None,
                "trace_id": trace_id,
            },
        )
        for key in candidates:
            quote = self._broker_quote_any(key)
            self.logger.info(
                "refresh_attempt",
                extra={
                    "symbol": symbol,
                    "key": key,
                    "ok": bool(quote),
                    "trace_id": trace_id,
                },
            )
            if not quote:
                # log each failed candidate at debug level; keep failure reason
                self._logger.debug(
                    "refresh candidate returned no quote",
                    extra={"symbol": symbol, "key": key, "trace_id": trace_id},
                )
                continue
            now_ms = self._now_ms()
            bid = _coerce_float(quote.get("bid"))
            ask = _coerce_float(quote.get("ask"))
            mid: float | None = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            cached_tick = dict(quote)
            with self._lock:
                self._latest_ticks[symbol] = cached_tick
                self._last_quote_ts_ms[symbol] = now_ms
                if mid is not None:
                    self._last_mid[symbol] = (mid, now_ms)
            self._notify_unified_manager(symbol, cached_tick)
            return dict(cached_tick)
        self._logger.error(
            "refresh_quote_failed_all_candidates",
            extra={
                "symbol": symbol,
                "candidates": candidates,
                "trace_id": trace_id,
            },
            exec_info=True,
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
        """Poll the broker for *symbol* and update cached ticks.

        Args:
            symbol: Trading symbol identifier requested for refresh.

        Returns:
            None.

        Raises:
            None.
        """

        candidates: list[str | int] = self._candidate_quote_keys(symbol)
        if not candidates:
            candidates = [symbol]
        quote: dict[str, Any] | None = None
        for key in candidates:
            broker_key: str | int
            if isinstance(key, int):
                broker_key = int(key)
            else:
                broker_key = key
            fetched = self._broker_quote_any(broker_key)
            if fetched:
                quote = fetched
                break
        if quote is None:
            return
        if not isinstance(quote, dict):
            quote = dict(quote)
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
                self._logger.error(
                    "WS subscribe skipped (no token)", extra={"symbol": symbol}
                )
                return
            self._logger.debug(
                "EVENT|subscribe|%s|token=%s|mode=full",
                symbol,
                token,
            )
            self._ws.subscribe_tokens([token], mode="full")
        except Exception as exc:  # noqa: BLE001
            # Log both message and details so it shows up even if the logger ignores
            # 'extra'.
            self._logger.error(
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

        self._logger.error(
            "Failed to resolve instrument token for %s",
            symbol,
            extra={"symbol": symbol},
        )
        return None

    # ------------------------------------------------------------------
    # Helpers

    def _coerce_from_depth(self, depth: Mapping[str, Any], side: str) -> float | None:
        """Safely extract best price from depth list without crashing."""
        try:
            if not depth:
                return None

            levels = depth.get(side)
            # Must be a list/tuple/iterable, but NOT a string or dict
            if not levels or not isinstance(levels, (list, tuple)):
                return None

            for entry in levels:
                if not isinstance(entry, Mapping):
                    continue

                # Check multiple common keys for price
                price = entry.get("price") or entry.get("p")
                if price:
                    try:
                        val = float(price)
                        if val > 0:
                            return val
                    except (ValueError, TypeError):
                        continue
        except Exception:
            # Swallow deep extraction errors to prevent tick stream crash
            return None
        return None

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
        """Normalize raw broker ticks into a standard internal format."""

        # 1. Extract LTP (Last Traded Price)
        # Priority: ltp -> last_price -> close -> last
        ltp = self._coerce_float(tick, "ltp", "last_price", "close", "last")

        # Deep search if top-level fails
        if (ltp is None or ltp <= 0) and isinstance(tick, Mapping):
            for value in tick.values():
                if isinstance(value, Mapping):
                    nested_ltp = self._coerce_float(value, "ltp", "last_price", "close")
                    if nested_ltp and nested_ltp > 0:
                        ltp = nested_ltp
                        break

        if ltp is None or ltp <= 0:
            return None

        # 2. Extract Bid/Ask
        bid = self._coerce_float(tick, "best_bid", "bid", "buy_price", "best_bid_price")
        ask = self._coerce_float(
            tick, "best_ask", "ask", "sell_price", "best_ask_price"
        )

        # 3. Fallback to Depth
        depth = tick.get("depth")
        if not isinstance(depth, Mapping):
            depth = {}

        if bid is None:
            bid = self._coerce_from_depth(depth, "buy")

        if ask is None:
            ask = self._coerce_from_depth(depth, "sell")

        # 4. Handle Empty Depth Gracefully (Prevent Errors)
        if bid is None:
            buy_levels = depth.get("buy")
            if isinstance(buy_levels, list) and not buy_levels:
                # Empty list = valid "no buyers" state
                self._logger.debug(
                    "Depth present but buy side empty", extra={"symbol": symbol}
                )
            # Fallback: Previous > LTP
            bid = previous.get("bid") if previous else ltp

        if ask is None:
            sell_levels = depth.get("sell")
            if isinstance(sell_levels, list) and not sell_levels:
                # Empty list = valid "no sellers" state
                self._logger.debug(
                    "Depth present but sell side empty", extra={"symbol": symbol}
                )
            # Fallback: Previous > LTP
            ask = previous.get("ask") if previous else ltp

        # Final Safety
        if bid is None:
            bid = ltp
        if ask is None:
            ask = ltp

        timestamp = self._coerce_timestamp(tick)

        normalized = {
            "symbol": symbol,
            "ltp": float(ltp),
            "last_price": float(ltp),
            "bid": float(bid),
            "ask": float(ask),
            "timestamp": timestamp,
            "depth": depth,
            "ltq": tick.get("last_quantity"),
            "oi": self._coerce_float(tick, "oi", "open_interest"),
        }

        # 5. Volume Handling
        volume = self._coerce_float(
            tick,
            "volume_traded_today",
            "volume",
            "volume_traded",
            "total_traded_volume",
        )
        if volume is None and previous:
            prev_vol = previous.get("volume")
            if isinstance(prev_vol, (int, float)):
                volume = float(prev_vol)

        if volume is not None:
            normalized["volume"] = float(volume)

        instrument_token = tick.get("instrument_token") or tick.get("token")
        if instrument_token is not None:
            normalized["instrument_token"] = instrument_token

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
        ex_ts = (
            tick.get("exchange_timestamp")
            or tick.get("last_trade_time")
            or tick.get("timestamp")
            or 0
        )

        volume = tick.get("volume_traded_today") or tick.get("volume") or 0
        oi = tick.get("oi") or tick.get("open_interest") or 0

        signature = (
            tick.get("ltp"),
            tick.get("bid"),
            tick.get("ask"),
            int(volume) // 100 if volume else 0,
            int(oi) // 100 if oi else 0,
            int(ex_ts) // 1000 if ex_ts else 0,
        )
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

    # -------------------------------------------------------------------------
    # ✅ "HUNTER-KILLER" FIX: Robust History Fetching
    # -------------------------------------------------------------------------
    async def fetch_history(
        self, symbol: str, interval: str, days: int = 3
    ) -> list[dict]:
        """
        Fetch historical data with Aggressive Fetcher Detection & Auto-Token Resolution.
        """
        # 1. Normalize Symbol
        symbol = symbol.strip().upper()

        # 2. Resolve Token (Try Cache -> Resolver -> Broker)
        token = getattr(self, "_token_by_symbol", {}).get(symbol)

        if not token:
            self._logger.info(
                f"🔎 History: Cache miss for {symbol}. Attempting force resolution..."
            )

            # Try Resolver
            resolver = getattr(self, "_resolver", None)
            if resolver:
                try:
                    # Try resolve() method
                    if hasattr(resolver, "resolve"):
                        t = resolver.resolve(symbol)
                        if t:
                            token = int(t)
                    # Try get_token() method (Fallback)
                    if not token and hasattr(resolver, "get_token"):
                        t = resolver.get_token(symbol)
                        if t:
                            token = int(t)
                except Exception:
                    pass

            # Try Broker Instrument Lookup (Final Fallback)
            if (
                not token
                and self._broker
                and hasattr(self._broker, "get_instrument_token")
            ):
                try:
                    t = self._broker.get_instrument_token(symbol)
                    if t:
                        token = int(t)
                except Exception:
                    pass

        if not token:
            self._logger.warning(
                f"❌ History Aborted: Could not resolve token for {symbol}"
            )
            return []

        # 3. Calculate Dates
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=days)

        # 4. Fetch Data (ROBUST SEARCH)
        if not self._broker:
            self._logger.error("❌ Broker instance is None.")
            return []

        try:
            fetcher = None

            # List of method names to hunt for
            candidates = [
                "historical_data",
                "get_historical_data",
                "history",
                "get_history",
            ]

            # A. Check Broker Direct
            for method in candidates:
                f = getattr(self._broker, method, None)
                if callable(f):
                    fetcher = f
                    break

            # B. Check Inner Client (kite/client)
            if not fetcher:
                client = getattr(
                    self._broker, "kite", getattr(self._broker, "client", None)
                )
                if client:
                    for method in candidates:
                        f = getattr(client, method, None)
                        if callable(f):
                            fetcher = f
                            break

            if callable(fetcher):
                # Run blocking I/O in thread
                data = await asyncio.to_thread(
                    fetcher, token, from_date, to_date, interval
                )
                if data:
                    self._logger.info(f"✅ Received {len(data)} candles for {symbol}")
                else:
                    self._logger.warning(
                        f"⚠️ History fetch returned 0 candles for {symbol} (Token: {token})"
                    )
                return data

            # Debugging Dump if failure persists
            self._logger.error(
                f"⚠️ Broker {type(self._broker).__name__} missing history capability. "
                f"Checked: {candidates}"
            )
            return []

        except Exception as e:
            self._logger.error(
                f"History fetch crashed for {symbol}: {e}", exc_info=True
            )
            return []


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
                "available_cash",
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
