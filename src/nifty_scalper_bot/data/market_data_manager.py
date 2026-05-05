"""Central market data manager responsible for tick fan-out and broker cache."""

from __future__ import annotations
from nifty_scalper_bot.core.message_bus import Message, MessageType

import asyncio
import inspect
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import logging
import math
import os
from random import uniform
import threading
import time
from typing import (
    Any,
    Callable,
    Deque,
    Iterable,
    Mapping,
    Sequence,
    cast,
)
import pandas as pd

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.data.candle_engine import CandleEngine
from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy
from nifty_scalper_bot.data.normalizers import normalize_history_row
from nifty_scalper_bot.data.validator import validate_tick
from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.streaming.websocket_manager import (
    ConnectionState,
    WebSocketManager,
)
from nifty_scalper_bot.utils.env import get_str
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger, get_tracer_logger, log_throttled
from nifty_scalper_bot.utils.market_hours import (
    MarketState,
    get_market_state,
    stale_threshold_for_symbol,
)
from nifty_scalper_bot.utils.metrics import Counter
from nifty_scalper_bot.utils.serialization import to_json_safe
from nifty_scalper_bot.utils.symbols import enforce_canonical, normalize_symbol

# NOTE: resolver is attached at runtime by app.py (ctx.market_data_manager._resolver).
# InstrumentManager is used as the resolver (Any type accepted).

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
class MarketSnapshot:
    """Normalized symbol snapshot for strategy quality gating."""

    symbol: str
    canonical_symbol: str
    ltp: float | None
    bid: float | None
    ask: float | None
    mid: float | None
    tick_age_s: float | None
    source: str
    real_ticks_last_60s: int
    latest_candle_provisional: bool
    latest_candle_synthetic: bool
    ohlc_valid: bool
    depth_available: bool = False
    bid_missing: bool = False
    ask_missing: bool = False
    bid_ask_source: str = "missing"
    tradable_quote: bool = False


@dataclass(slots=True)
class ResolvedInstrument:
    """Resolved instrument details. Args: symbol. Returns: normalized payload. Raises: none."""

    canonical_symbol: str
    broker_symbol: str
    instrument_token: int | None
    exchange: str
    tradingsymbol: str




@dataclass(frozen=True)
class PollingPolicy:
    """Polling policy values. Args: settings. Returns: policy. Raises: none."""

    context_stale_seconds: float = 30.0
    option_stale_seconds: float = 60.0
    interval_seconds: float = 1.0
    max_symbols: int = 12
    error_log_throttle_seconds: float = 30.0

    @classmethod
    def from_settings(cls, settings: Any | None) -> "PollingPolicy":
        """Build policy from settings. Args: settings. Returns: policy. Raises: none."""
        if settings is None:
            return cls()
        return cls(
            context_stale_seconds=float(getattr(settings, "poll_context_stale_seconds", 30.0)),
            option_stale_seconds=float(getattr(settings, "poll_option_stale_seconds", 60.0)),
            interval_seconds=float(getattr(settings, "poll_interval_seconds", 1.0)),
            max_symbols=int(getattr(settings, "poll_max_symbols", 12)),
            error_log_throttle_seconds=float(getattr(settings, "poll_error_log_throttle_seconds", 30.0)),
        )
def canonical_symbol(symbol: str) -> str:
    """Canonicalize symbols (including NIFTY spot aliases) to EXCHANGE:SYMBOL."""
    normalized = enforce_canonical(normalize_symbol(str(symbol or "")))
    if normalized == "NSE:NIFTY 50":
        return "NSE:NIFTY"
    return normalized


NIFTY_SPOT_TOKEN = 256265

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
        self._external_tick_handler: Callable[[dict[str, Any]], None] | None = None
        self._broker = broker
        self._rest_client = broker
        self._websocket = websocket
        # FIX: Explicitly assign self._ws for internal use
        self._ws = websocket
        self._settings = settings or {}
        self._md_policy = MarketDataPolicy.from_env()
        self._resolver = resolver
        self._logger = get_logger(__name__)
        # FIX: Initialize cache_len before it is used
        self._cache_len = cache_len

        # FIX: Initialize duplicate window (Missing in your file, causing the crash)
        self._duplicate_window = self._parse_float_env(
            "MDM_DUPLICATE_WINDOW_SEC", default=1.0, minimum=0.0
        )

        self._subscribers: dict[str, set[TickCallback]] = defaultdict(set)
        self._bar_subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._latest_ticks: dict[str, dict[str, Any]] = {}
        self._history: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._cache_len)
        )
        self._token_by_symbol: dict[str, int] = {}
        self._symbol_by_token: dict[int, str] = {}
        self._desired_tokens: set[int] = set()
        self._symbol_to_token: dict[str, int] = {}
        self._token_to_symbol: dict[int, str] = {}
        self._token_by_symbol["NSE:NIFTY"] = 256265       # canonical
        self._symbol_by_token[256265] = "NSE:NIFTY"
        self._token_by_symbol["NSE:BANKNIFTY"] = 260105
        self._symbol_by_token[260105] = "NSE:BANKNIFTY"
        self._symbol_to_token.update(self._token_by_symbol)
        self._token_to_symbol.update(self._symbol_by_token)
        self._last_signature: dict[
            str, tuple[tuple[float | None, float | None, float | None], float]
        ] = {}
        self._last_tick_wallclock: dict[str, float] = {}
        self._last_quote_ts_ms: dict[str, float] = {}
        self._last_mid: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._authoritative_tick_lock = threading.Lock()
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
        self._last_ws_tick_mono: float = 0.0
        self._event_loop_lag_seconds: float = 0.0
        self._hydration_status: dict[str, str] = {}
        self._hydration_log_ts: dict[str, float] = {}
        self._last_hb_mono: float | None = None
        self._heartbeat_callbacks: list[Callable[[float], None]] = []
        self._fallback_enabled = False
        self.bus = None
        self._poll_jitter_pct = 0.0
        self._poll_batch_ceiling = 0
        self._ohlc: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._cache_len)
        )
        self._engines: dict[str, CandleEngine] = {}
        self._last_historical_ts: dict[str, float] = {}
        self._last_tick_ts: dict[str, float] = {}
        self._last_tick_snapshot: dict[str, dict[str, Any]] = {}
        self._readiness_requirements: dict[str, Any] = {}
        self._last_readiness_state: dict[str, Any] = {
            "hard_ready": False,
            "spot_ready": False,
            "missing_hard": [],
        }
        self._quote_api_available = True
        self._quote_api_error: str | None = None
        self._quote_api_last_checked_at: float | None = None
        self._spot_ready_logged = False
        self._spot_refresh_last_attempt_mono: float = 0.0
        # Bounded queue: drops oldest when full so a stall in the async
        # consumer never grows memory without bound. Size 10 000 is ~8 s
        # of headroom at 1 000 tps.
        self._tick_queue_maxsize = self._parse_int_env(
            "MDM_TICK_QUEUE_MAX", default=10_000, minimum=1_000
        )
        self._tick_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._tick_queue_maxsize
        )
        self._tick_queue_dropped = 0
        self._last_tick_queue_full_log = 0.0
        self._tick_consumer_task: asyncio.Task[None] | None = None
        # Fallback worker thread for when no asyncio loop is registered
        # (e.g. early boot, polling-only mode).  Isolates the WS thread
        # from any processing work — the WS callback must only enqueue.
        self._tick_worker_stop = threading.Event()
        self._tick_worker_thread: threading.Thread | None = None
        self._account_snapshot: dict[str, float] = {}
        self._account_updated_at: float = 0.0
        self._tracked_symbols: set[str] = set()
        self._active_subscribed_symbols: set[str] = set()
        self._symbols_with_tick: set[str] = set()
        self._ticks_received_per_symbol: dict[str, int] = defaultdict(int)
        self._tick_stats: dict[str, int] = defaultdict(int)
        self._last_tick_stats_log = time.monotonic()
        self._last_tick_rate_snapshot_at = time.monotonic()
        self._last_tick_rate_snapshot: dict[str, int] = {}
        self._account_cache_ttl = self._parse_float_env(
            "MDM_ACCOUNT_CACHE_TTL", default=60.0, minimum=1.0
        )
        self._account_segment = _resolve_account_segment()

        self._margin_lock = threading.RLock()
        self._margin_snapshot: dict[str, Any] | None = None
        self._last_margin_refresh: float = 0.0
        self.latest_ticks: list[dict[str, Any]] = []
        self.last_tick_time = time.time()
        self._stale_threshold_seconds = self._parse_float_env(
            "MDM_STALE_THRESHOLD_SECONDS", default=10.0, minimum=1.0
        )
        self._critical_symbol_stale_seconds = self._parse_float_env(
            "MDM_CRITICAL_SYMBOL_STALE_SECONDS", default=30.0, minimum=1.0
        )
        self._option_stale_seconds = self._parse_float_env(
            "MDM_OPTION_STALE_SECONDS", default=60.0, minimum=1.0
        )
        self._polling_policy = PollingPolicy.from_settings(settings)
        self._poll_error_last_log: dict[str, float] = {}
        self._history_lock = asyncio.Lock()
        self._last_history_request_ts = 0.0
        self._history_min_interval_sec = float(
            getattr(settings, "history_min_interval_sec", 1.2)
            if settings is not None
            else 1.2
        )
        self._logger.info(
            "POLL_POLICY_READY context_stale=%s option_stale=%s interval=%s max_symbols=%s",
            self._polling_policy.context_stale_seconds,
            self._polling_policy.option_stale_seconds,
            self._polling_policy.interval_seconds,
            self._polling_policy.max_symbols,
            extra={"event": "POLL_POLICY_READY"},
        )
        self._poll_fallback_count = 0
        self._ltp_stale_seconds = self._parse_float_env(
            "MDM_LTP_STALE_SECONDS", default=5.0, minimum=1.0
        )
        self._index_ltp_stale_seconds = self._parse_float_env(
            "MDM_INDEX_LTP_STALE_SECONDS", default=120.0, minimum=1.0
        )
        self._offmarket_index_ltp_stale_seconds = self._parse_float_env(
            "MDM_OFFMARKET_INDEX_LTP_STALE_SECONDS", default=3600.0, minimum=60.0
        )
        self._option_ltp_stale_seconds = self._parse_float_env(
            "MDM_OPTION_LTP_STALE_SECONDS", default=30.0, minimum=1.0
        )
        self._future_ltp_stale_seconds = self._parse_float_env(
            "MDM_FUTURE_LTP_STALE_SECONDS", default=60.0, minimum=1.0
        )
        self._generic_ltp_stale_seconds = self._parse_float_env(
            "MDM_GENERIC_LTP_STALE_SECONDS", default=60.0, minimum=1.0
        )
        self._ltp_stale_warn_last: dict[str, float] = {}
        self._spot_ws_first_tick_seen = False
        self._spot_ws_first_tick_seen_logged = False
        self._last_spot_resubscribe_attempt = 0.0
        self._last_spot_stale_log = 0.0
        self._pending_subscription_tokens: set[int] = set()
        self._pending_subscriptions: set[int] = set()
        self._dispatched_subscriptions: set[int] = set()
        self._confirmed_subscriptions: set[int] = set()
        self._last_rest_refresh_attempt: dict[str, float] = {}
        self._tick_warn_last: dict[str, float] = (
            {}
        )  # ✅ FIX: rate-limit cache-miss warnings
        self._ws_raw_tick_log_at: dict[str, float] = {}
        self._ws_stored_tick_log_at: dict[str, float] = {}
        self._seed_attempt_last: dict[str, float] = {}
        self._seed_completed = False
        self._seeded_symbols: set[str] = set()
        self.ready: bool = False
        self.degraded: bool = False
        # True once at least one symbol has successfully passed the readiness
        # gate.  The DEGRADED mode transition in wait_until_ready() is only
        # armed after hydration has completed at least once so that a cold
        # startup that never receives data does not immediately enter DEGRADED.
        self.hydration_complete: bool = False
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

        ws_disabled = os.getenv("WEBSOCKET__DISABLED", "false").lower() == "true"
        poll_src = os.getenv("DATA_READINESS_SOURCE", "").upper() == "POLL"
        poll_fallback_env = os.getenv("MDM_POLL_FALLBACK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._rest_poll_enabled = ws_disabled or poll_src or poll_fallback_env
        self._rest_poll_interval = self._parse_float_env(
            "MDM_POLL_INTERVAL_SECONDS", default=1.5, minimum=0.5
        )
        configured_poll_max = self._parse_int_env(
            "MDM_POLL_MAX_SYMBOLS", default=50, minimum=1
        )
        self._rest_poll_max_symbols = (
            configured_poll_max if self._rest_poll_enabled else 0
        )
        self._fallback_enabled = self._rest_poll_enabled
        self._poll_bar_state: dict[str, dict[str, float | None | int]] = {}
        self._last_poll_bucket: dict[str, int] = {}
        self._rest_poll_stop = threading.Event()
        self._rest_poll_thread: threading.Thread | None = None
        self._health_monitor_stop = threading.Event()
        self._health_monitor_thread: threading.Thread | None = None
        self._zombie_tick_threshold_sec = self._parse_float_env(
            "ZOMBIE_TICK_THRESHOLD_SEC", default=60.0, minimum=10.0
        )
        self._zombie_restart_failures = 0
        self._zombie_restart_attempts: Deque[float] = deque()
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
        # Exponential-backoff ceiling for consecutive zombie restarts.
        # Each successive attempt doubles the cooldown (with full
        # jitter) up to this ceiling; success resets it.
        self._zombie_restart_cooldown_max_sec = self._parse_float_env(
            "ZOMBIE_RESTART_COOLDOWN_MAX_SEC", default=300.0, minimum=30.0
        )
        self._zombie_restart_consecutive = 0
        self._zombie_stale_logged = False
        self._rest_refresh_inflight: set[str] = set()
        self._polling_fallback_streamer: Any | None = None
        self._tick_stale_threshold_ms = self._parse_int_env(
            "TICK_STALE_MS", default=2_000, minimum=0
        )
        self._min_required_bars = self._parse_int_env(
            "MIN_INDICATOR_BARS", default=20, minimum=1
        )

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

        # NOTE: WebSocketManager._on_ticks already invokes
        # market_data_manager.process_ticks(ticks) on every batch (see
        # streaming/websocket_manager.py::_on_ticks).  Previously MDM
        # *also* installed self._enqueue_tick_threadsafe as the per-tick
        # _on_tick_callback which caused every tick to be enqueued
        # twice (the duplicates were silently dropped by the
        # non-monotonic timestamp filter in _process_queued_tick, but
        # they still consumed queue slots and CPU).  We no longer reach
        # for the callback slot here — process_ticks is the single
        # ingress for WS ticks.
        self._m_ticks = Counter("mdm_ticks_total", "Normalized ticks processed")
        self._last_balance_log_time = 0.0
        self._started = False
        if self._ws is not None and hasattr(self._ws, "on_tick"):
            try:
                self._ws.on_tick = self._on_tick
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("websocket on_tick binding skipped: %s", exc)

    def _initialize_instruments(self) -> None:
        """Load instrument token mappings from broker with transient retry.

        Transient Zerodha REST failures on boot used to leave the
        token↔symbol map empty, which then cascaded into "Token missing
        for NSE:NIFTY" errors and DEGRADED mode.  We now retry each
        exchange up to 3 times with exponential backoff before giving
        up, and a failure to load ANY exchange is loudly logged.

        Args: none. Returns: None. Raises: None.
        """

        if self._broker is None or not hasattr(self._broker, "instruments"):
            return
        loaded = 0
        any_success = False
        # Load NFO (derivatives) first, then NSE (index tokens like
        # NIFTY 50). Previously the call had no exchange argument which
        # defaulted to NSE, leaving all NFO futures/options tokens
        # missing from the mapping.
        for exchange in ("NFO", "NSE"):
            exchange_instruments: Any = None
            for attempt in range(3):
                try:
                    exchange_instruments = self._broker.instruments(exchange)
                    break
                except Exception as exc:  # noqa: BLE001
                    delay = min(8.0, 2.0 ** attempt)
                    self._logger.warning(
                        "instrument load %s attempt %d failed: %s — "
                        "retry in %.1fs",
                        exchange,
                        attempt + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            if not exchange_instruments:
                self._logger.error(
                    "Could not load instruments for %s after 3 attempts",
                    exchange,
                    extra={"event": "mdm_instrument_load_failed", "exchange": exchange},
                )
                continue
            any_success = True
            with self._lock:
                for ins in exchange_instruments:
                    try:
                        token = int(ins["instrument_token"])
                        symbol = str(ins["tradingsymbol"]).strip()
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not symbol:
                        continue
                    canonical_symbol = normalize_symbol(symbol)
                    try:
                        self.register_symbol(canonical_symbol, token)
                        loaded += 1
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "register_symbol skipped %s (%s): %s",
                            canonical_symbol,
                            token,
                            exc,
                        )
        if not any_success:
            # Downstream zombie checker + polling fallback will still
            # work for explicitly-registered symbols (NIFTY /
            # BANKNIFTY, plus anything resolver injects at runtime).
            self._logger.error(
                "Instrument load failed for all exchanges — running with "
                "seeded token map only",
                extra={"event": "mdm_instrument_load_empty"},
            )
        self._logger.info(
            f"Loaded {loaded} instruments (NFO + NSE)",
            extra={"event": "mdm_instruments_loaded", "count": loaded},
        )

    def validate_token_symbol_mappings(self) -> None:
        """Args: none; Returns: None; Raises: RuntimeError on mapping mismatch."""

        with self._lock:
            symbol_by_token = dict(self._symbol_by_token)
            token_by_symbol = dict(self._token_by_symbol)
        mismatches: list[str] = []
        for token, symbol in symbol_by_token.items():
            reverse = token_by_symbol.get(symbol)
            if reverse != token:
                mismatches.append(f"{symbol}:{token}->{reverse}")
        if mismatches:
            raise RuntimeError(
                f"Token/symbol mapping mismatch detected ({len(mismatches)}): {mismatches[:5]}"
            )

    def ingest_rest_quote(self, symbol: str, quote: Mapping[str, Any]) -> None:
        """Ingest a REST/polling quote for symbol.

        Used by _rest_poll_loop as a WebSocket fallback.  DataHub.ingest_tick
        deduplicates poll ticks against live WS ticks, so this will be
        suppressed automatically when WebSocket is healthy.

        Args: symbol, quote. Returns: None. Raises: None.
        """
        if not symbol or not quote:
            return
        try:
            symbol = normalize_symbol(str(symbol))
            source = str(quote.get("source") or "poll").lower()
            prepared = self._prepare_rest_tick(quote, source=source)
            with self._lock:
                previous = self._latest_ticks.get(symbol)
            normalized = self._normalize_tick(symbol, prepared, previous)
            if normalized is None:
                return
            poll_ts = normalized.get("timestamp")
            if isinstance(poll_ts, (int, float)):
                poll_bucket = int(float(poll_ts) // 60)
                current_live_bucket = int(time.time() // 60)
                if poll_bucket > current_live_bucket:
                    return
            if self._is_duplicate(symbol, normalized):
                self._store_tick(symbol, normalized)
            else:
                self._emit_tick(symbol, normalized, source=source)
            self._process_poll_quote(symbol, normalized)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "ingest_rest_quote failed for %s: %s", symbol, exc
            )

    def _process_poll_quote(self, symbol: str, quote: Mapping[str, Any]) -> None:
        """Aggregate poll quotes into minute candles. Args: symbol, quote. Returns: None. Raises: Exception."""
        try:
            ts = float(quote.get("timestamp") or time.time())
            if ts > 1e12:
                ts /= 1000.0

            bucket = int(ts // 60)
            state = self._poll_bar_state.setdefault(
                symbol,
                {
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": 0,
                },
            )

            price = float(quote.get("last_price") or quote.get("ltp") or 0.0)
            if price <= 0:
                return

            if state["open"] is None:
                state["open"] = price
                state["high"] = price
                state["low"] = price
            else:
                state["high"] = max(float(state["high"] or price), price)
                state["low"] = min(float(state["low"] or price), price)
            state["close"] = price

            last_bucket = self._last_poll_bucket.get(symbol)
            if last_bucket is None:
                self._last_poll_bucket[symbol] = bucket
                return

            if bucket > last_bucket:
                candle = {
                    "symbol": symbol,
                    "open": state["open"],
                    "high": state["high"],
                    "low": state["low"],
                    "close": state["close"],
                    "volume": int(state["volume"] or 0),
                    "timestamp": last_bucket * 60,
                    "source": "poll",
                }
                self._poll_bar_state[symbol] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                }
                self._last_poll_bucket[symbol] = bucket
                self._emit_poll_candle(candle)
                self._logger.info("POLL CANDLE EMITTED: %s", symbol)
        except Exception as e:
            self._logger.error("Poll candle build failed for %s: %s", symbol, e)

    def _emit_poll_candle(self, candle: Mapping[str, Any]) -> None:
        """Ingest poll-built candle into MDM historical buffers.

        Args: candle mapping.
        Returns: None.
        Raises: None.
        """
        try:
            self.ingest_historical_bar(dict(candle))
        except Exception as e:
            self._logger.error("Poll candle emit failed: %s", e)

    def ingest_historical_bar(self, bar: Mapping[str, Any]) -> None:
        """Ingest one historical OHLC bar into MDM-owned state.

        Args: bar payload containing symbol/open/high/low/close/volume/timestamp.
        Returns: None.
        Raises: None.
        """
        try:
            symbol = normalize_symbol(str(bar.get("symbol") or ""))
            if not symbol:
                return
            ts = bar.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            payload = {
                "symbol": symbol,
                "open": float(bar.get("open", 0.0) or 0.0),
                "high": float(bar.get("high", 0.0) or 0.0),
                "low": float(bar.get("low", 0.0) or 0.0),
                "close": float(bar.get("close", 0.0) or 0.0),
                "volume": int(float(bar.get("volume", 0) or 0)),
                "timestamp": ts.isoformat(),
                "source": "historical",
            }
            self._history[symbol].append(dict(payload))
            self._ohlc[self._bar_symbol_key(symbol)].append(dict(payload))
            if isinstance(ts, datetime):
                self._last_historical_ts[symbol] = ts.timestamp()
            self.update_hydration_status(symbol, self._ohlc[self._bar_symbol_key(symbol)])
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in ingest_historical_bar: %s", exc, exc_info=exc)

    def ingest_historical_ohlc(
        self, symbol: str, bars: Sequence[Mapping[str, Any]]
    ) -> int:
        """Ingest historical OHLC bars into MDM and return accepted count.

        Args: symbol, bars.
        Returns: accepted bar count.
        Raises: None.
        """
        accepted = 0
        try:
            for row in bars:
                data = dict(row)
                data["symbol"] = symbol
                self.ingest_historical_bar(data)
                accepted += 1
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in ingest_historical_ohlc: %s", exc, exc_info=exc)
        return accepted

    def subscribe_bars(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Args: callback. Returns: None. Raises: None."""
        if callback not in self._bar_subscribers:
            self._bar_subscribers.append(callback)

    def _publish_closed_bar(self, bar: dict[str, Any]) -> None:
        """Args: bar. Returns: None. Raises: None."""
        for cb in list(self._bar_subscribers):
            try:
                cb(dict(bar))
            except Exception as exc:
                symbol = str(bar.get("symbol") or "")
                callback_name = getattr(cb, "__qualname__", repr(cb))
                log_throttled(
                    self._logger,
                    f"bar_subscriber_failed:{symbol}:{callback_name}:{type(exc).__name__}",
                    "BAR_SUBSCRIBER_FAILED symbol=%s callback=%s error=%s"
                    % (symbol, callback_name, exc),
                    interval_sec=60,
                    level=logging.ERROR,
                    exc_info=True,
                    extra={
                        "event": "BAR_SUBSCRIBER_FAILED",
                        "symbol": symbol,
                        "callback": callback_name,
                        "error": str(exc),
                    },
                )

    def set_readiness_requirements(
        self,
        *,
        spot_symbol: str,
        futures_symbol: str,
        atm_ce_symbol: str | None,
        atm_pe_symbol: str | None,
        option_symbols: Sequence[str],
    ) -> None:
        """Set explicit required-symbol/quorum readiness rules.

        Args: canonical symbols and option set.
        Returns: None.
        Raises: None.
        """
        self._readiness_requirements = {
            "spot": normalize_symbol(spot_symbol),
            "futures": normalize_symbol(futures_symbol),
            "atm_ce": normalize_symbol(atm_ce_symbol or "") if atm_ce_symbol else None,
            "atm_pe": normalize_symbol(atm_pe_symbol or "") if atm_pe_symbol else None,
            "options": [normalize_symbol(s) for s in option_symbols if s],
        }

    def set_unified_manager(self, manager: Any | None) -> None:
        """Retain compatibility hook. Args: manager. Returns: None. Raises: None."""
        self._unified_manager = manager

    @staticmethod
    def _bar_symbol_key(symbol: str) -> str:
        """Return normalized bar key for *symbol*."""

        return normalize_symbol(symbol)

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
        """Robust option chain fetcher using direct broker fallback."""
        normalized_underlying = (underlying or "").strip().upper()
        
        # FIX: Directly ask broker if resolver is failing or in-memory cache is empty
        option_contracts = []
        try:
            # First try the existing resolver logic
            if self._resolver:
                option_contracts = self._resolver.option_contracts(normalized_underlying)
            
            # CRITICAL FALLBACK: If resolver returns nothing, query broker for active NFO instruments
            if not option_contracts and hasattr(self._broker, "instruments"):
                self._logger.info(f"Resolver failed. Fetching NIFTY chain directly from broker for {expiry}")
                all_nfo = self._broker.instruments("NFO")
                option_contracts = [
                    ins for ins in all_nfo 
                    if ins["name"] == normalized_underlying 
                    and ins["instrument_type"] in ("CE", "PE")
                    and str(ins["expiry"]) == expiry
                ]
        except Exception as exc:
            self._logger.error(f"Option chain recovery failed: {exc}")
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
    def start(self, *, defer_ws: bool = False) -> None:
        """Start the market data manager.

        Args:
            defer_ws: When ``True`` the WebSocket transport is **not** started
                here.  Callers must invoke :meth:`start_websocket` later once
                the token universe has been resolved — this guarantees the
                initial ``_on_connect`` handshake observes a populated
                ``_tokens`` set instead of subscribing with the single spot
                token only.
        """

        if self._started:
            if not defer_ws:
                self.start_websocket()
            return
        self._started = True
        self._initialize_instruments()
        if not defer_ws:
            self.start_websocket()
        self._ensure_tick_consumer(reason="mdm_start")
        self._start_health_monitor()
        if self._rest_poll_enabled:
            self._start_rest_poll()

    def start_websocket(self) -> None:
        """Start the websocket transport if present.

        Separated from :meth:`start` so that callers can first populate the
        token universe (spot + futures + ATM options) and then bring the WS
        online — this avoids the misleading ``tokens=1`` banner at connect
        time and ensures the very first subscribe already covers every token
        of interest.
        """

        if self._ws is None:
            return
        status = self.ws_status_snapshot()
        if bool(getattr(self, "_ws_start_requested", False)) and bool(
            status.get("connect_task_alive")
        ):
            return
        self._ws_start_requested = True
        try:
            self._reconcile_ws_subscriptions()
            desired_tokens = len(self._desired_tokens)
            ws_token_count = int(status.get("tokens", 0))
            self._logger.info(
                "MDM_WS_START_REQUESTED desired_token_count=%s ws_token_count=%s",
                desired_tokens,
                ws_token_count,
                extra={
                    "event": "MDM_WS_START_REQUESTED",
                    "desired_token_count": desired_tokens,
                    "ws_token_count": ws_token_count,
                },
            )
            self._ws.start()
            post_status = self.ws_status_snapshot()
            self._ws_connect_task_started = bool(post_status.get("connect_task_alive"))
            self._ws_connected = bool(post_status.get("connected"))
            if not self._ws_connected:
                self._logger.info(
                    "MDM_WS_START_PENDING_CONNECTION connect_task_alive=%s desired_token_count=%s ws_token_count=%s",
                    self._ws_connect_task_started,
                    len(self._desired_tokens),
                    post_status.get("tokens"),
                    extra={"event": "MDM_WS_START_PENDING_CONNECTION", "connect_task_alive": self._ws_connect_task_started, "desired_token_count": len(self._desired_tokens), "ws_token_count": post_status.get("tokens")},
                )
            if not self._ws_connect_task_started and not self._ws_connected:
                self._ws_start_requested = False
                self._logger.warning(
                    "MDM_WS_START_NO_CONNECT_TASK desired_token_count=%s ws_token_count=%s",
                    len(self._desired_tokens),
                    post_status.get("tokens"),
                    extra={"event": "MDM_WS_START_NO_CONNECT_TASK", "desired_token_count": len(self._desired_tokens), "ws_token_count": post_status.get("tokens")},
                )
        except Exception:
            self._ws_start_requested = False
            self._logger.exception("Failure in ws.start")

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._ws is not None:
            try:
                self._ws.stop()
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("ws.stop() raised: %s", exc)
        if self._tick_consumer_task is not None:
            try:
                self._tick_consumer_task.cancel()
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("tick consumer cancel raised: %s", exc)
            self._tick_consumer_task = None
        if self._tick_worker_thread is not None:
            self._tick_worker_stop.set()
            self._tick_worker_thread.join(timeout=2.0)
            self._tick_worker_thread = None
            self._tick_worker_stop.clear()
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
        self._ws_connect_task_started = bool(
            self.ws_status_snapshot().get("connect_task_alive")
        )
        self._ws_started = bool(connected)
        if not connected and not self._ws_connect_task_started:
            self._ws_start_requested = False
        if self._ws_connected:
            pending_tokens = sorted(
                self._pending_subscriptions or self._pending_subscription_tokens
            )
            if pending_tokens:
                self._logger.info(
                    "WS_PENDING_SUBSCRIPTION_FLUSH count=%s",
                    len(pending_tokens),
                    extra={
                        "event": "WS_PENDING_SUBSCRIPTION_FLUSH",
                        "count": len(pending_tokens),
                    },
                )
            self._reconcile_ws_subscriptions()
            self._pending_subscription_tokens.clear()
            self._pending_subscriptions.clear()

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
    def _canonical_symbol(self, symbol: str) -> str:
        requested = str(symbol or "")
        value = canonical_symbol(requested)
        if value.count(":") != 1:
            raise RuntimeError(f"Malformed canonical symbol: {value}")
        if requested.strip().upper() != value:
            self._logger.info(
                "SYMBOL_CANONICALIZED requested=%s canonical=%s",
                requested,
                value,
                extra={
                    "event": "SYMBOL_CANONICALIZED",
                    "requested": requested,
                    "canonical": value,
                },
            )
        return value

    def _resolve_instrument(self, symbol: str) -> ResolvedInstrument:
        """Resolve canonical and broker symbols. Args: symbol; Returns: resolved instrument; Raises: none."""
        canonical_value = self._canonical_symbol(symbol)
        exchange, tradingsymbol = canonical_value.split(":", 1)
        broker_symbol = canonical_value
        token = self._symbol_to_token.get(canonical_value) or self._token_by_symbol.get(
            canonical_value
        )
        resolver = getattr(self, "_resolver", None)
        if resolver is not None and hasattr(resolver, "resolve_symbol"):
            try:
                resolved = resolver.resolve_symbol(canonical_value)
                if resolved:
                    broker_symbol = str(
                        resolved.get("broker_symbol")
                        or resolved.get("exchange_symbol")
                        or f"{resolved.get('exchange', exchange)}:{resolved.get('tradingsymbol', tradingsymbol)}"
                    )
                    token = int(
                        resolved.get("instrument_token")
                        or resolved.get("token")
                        or token
                        or 0
                    ) or None
                    exchange = str(resolved.get("exchange") or exchange)
                    tradingsymbol = str(resolved.get("tradingsymbol") or tradingsymbol)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in _resolve_instrument: %s", exc)
        if canonical_value == "NSE:NIFTY":
            if token is None:
                self._logger.error(
                    "SPOT_TOKEN_RESOLUTION_FAILED symbol=NSE:NIFTY reason=token_not_found"
                )
            else:
                self._logger.info(
                    "SPOT_BROKER_SYMBOL_RESOLVED canonical=NSE:NIFTY broker_symbol=%s token=%s",
                    broker_symbol,
                    token,
                )
        return ResolvedInstrument(
            canonical_symbol=canonical_value,
            broker_symbol=broker_symbol,
            instrument_token=token,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
        )

    def _reconcile_ws_subscriptions(self) -> None:
        """Apply desired token set to websocket transport. Args: none. Returns: none. Raises: none."""

        ws = self._ws
        if ws is None or not hasattr(ws, "set_tokens"):
            return

        try:
            changed = ws.set_tokens(sorted(self._desired_tokens))
            self._dispatched_subscriptions.update(self._desired_tokens)
            connected = self._is_ws_connected()
            self._logger.debug(
                "ws_tokens_reconciled desired=%d changed=%s connected=%s",
                len(self._desired_tokens),
                changed,
                connected,
                extra={
                    "event": "ws_tokens_reconciled",
                    "desired_tokens": len(self._desired_tokens),
                    "changed": changed,
                    "connected": connected,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("ws token reconcile deferred: %s", exc)

        if hasattr(ws, "is_connected") and not self._is_ws_connected():
            self._pending_subscription_tokens.update(self._desired_tokens)
            self._pending_subscriptions.update(self._desired_tokens)
            return

    def request_token_subscription(
        self,
        token: int,
        symbol: str | None = None,
    ) -> bool:
        """Ensure token subscription intent is recorded and reconciled. Args: token/symbol. Returns: success flag. Raises: none."""
        try:
            token_int = int(token)
        except Exception:
            return False
        if token_int <= 0:
            return False
        normalized_symbol: str | None = None
        if symbol:
            normalized_symbol = self._canonical_symbol(symbol)
            self.register_symbol(normalized_symbol, token_int)

        with self._lock:
            self._desired_tokens.add(token_int)
            if normalized_symbol:
                self._symbol_to_token[normalized_symbol] = token_int
                self._token_to_symbol[token_int] = normalized_symbol
                self._symbol_by_token[token_int] = normalized_symbol
        self._reconcile_ws_subscriptions()
        return True

    def request_token_subscriptions(self, tokens: Iterable[int]) -> int:
        """Record token subscription intents. Args: tokens. Returns: number of newly added tokens. Raises: none."""

        normalized: set[int] = set()
        for token in tokens:
            try:
                token_int = int(token)
            except Exception:
                continue
            if token_int > 0:
                normalized.add(token_int)
        if not normalized:
            return 0
        with self._lock:
            before = len(self._desired_tokens)
            self._desired_tokens.update(normalized)
            added_count = len(self._desired_tokens) - before
        self._reconcile_ws_subscriptions()
        return added_count

    def request_token_unsubscriptions(self, tokens: Iterable[int]) -> int:
        """Record token removal intents. Args: tokens. Returns: number of removed tokens. Raises: none."""

        normalized: set[int] = set()
        for token in tokens:
            try:
                token_int = int(token)
            except Exception:
                continue
            if token_int > 0:
                normalized.add(token_int)
        if not normalized:
            return 0
        with self._lock:
            before = len(self._desired_tokens)
            self._desired_tokens.difference_update(normalized)
            removed_count = before - len(self._desired_tokens)
            for token in normalized:
                mapped_symbol = self._token_to_symbol.pop(token, None)
                if mapped_symbol is not None:
                    current = self._symbol_to_token.get(mapped_symbol)
                    if current == token:
                        self._symbol_to_token.pop(mapped_symbol, None)
        if removed_count > 0:
            self._reconcile_ws_subscriptions()
        return removed_count

    def request_token_unsubscription(
        self,
        token: int,
        symbol: str | None = None,
    ) -> bool:
        """Record token removal intent. Args: token/symbol. Returns: changed flag. Raises: none."""

        token_int = int(token)
        if token_int <= 0:
            return False
        normalized_symbol: str | None = None
        if symbol:
            normalized_symbol = self._canonical_symbol(symbol)

        changed = False
        with self._lock:
            if token_int in self._desired_tokens:
                self._desired_tokens.discard(token_int)
                changed = True
            if normalized_symbol:
                current = self._symbol_to_token.get(normalized_symbol)
                if current == token_int:
                    self._symbol_to_token.pop(normalized_symbol, None)
                    changed = True
            mapped_symbol = self._token_to_symbol.pop(token_int, None)
            if mapped_symbol is not None:
                current = self._symbol_to_token.get(mapped_symbol)
                if current == token_int:
                    self._symbol_to_token.pop(mapped_symbol, None)
                changed = True
        if changed:
            self._reconcile_ws_subscriptions()
        return changed

    def desired_token_count(self) -> int:
        """Return desired token count. Args: none. Returns: token count. Raises: none."""

        with self._lock:
            return len(self._desired_tokens)

    def desired_tokens_snapshot(self) -> list[int]:
        """Return sorted desired tokens. Args: none. Returns: tokens list. Raises: none."""

        with self._lock:
            return sorted(self._desired_tokens)

    def ws_token_count(self) -> int | None:
        """Return websocket token count if websocket exists. Args: none. Returns: token count or None. Raises: none."""
        ws = self._ws
        if ws is None:
            return None
        tokens = getattr(ws, "_tokens", None)
        if tokens is None:
            return None
        try:
            return len(tokens)
        except Exception:
            return None

    def request_symbol_unsubscription(self, symbol: str) -> bool:
        """Record symbol removal intent. Args: symbol. Returns: changed flag. Raises: none."""

        normalized = self._canonical_symbol(symbol)
        token = self._symbol_to_token.get(normalized) or self._token_by_symbol.get(normalized)
        if token is None:
            return False
        return self.request_token_unsubscription(token, symbol=normalized)

    def request_symbol_subscription(self, symbol: str) -> bool:
        """Record symbol subscription intent. Args: symbol. Returns: changed flag. Raises: none."""

        normalized = self._canonical_symbol(symbol)
        token = self._resolve_token(normalized)
        if token is None:
            return False
        return self.request_token_subscription(token, symbol=normalized)

    def request_symbol_subscriptions(self, symbols: Iterable[str]) -> int:
        """Record symbol subscription intents. Args: symbols. Returns: newly added token count. Raises: none."""

        changed = 0
        for symbol in symbols:
            if self.request_symbol_subscription(symbol):
                changed += 1
        return changed

    def disable_rest_polling(self, *, reason: str | None = None) -> None:
        """Disable the background REST poller when another component owns fallback."""

        self._rest_poll_enabled = False
        self._fallback_enabled = False
        self._rest_poll_max_symbols = 0
        if self._rest_poll_thread is not None:
            self._rest_poll_stop.set()
            self._rest_poll_thread.join(timeout=2.0)
            self._rest_poll_thread = None
            self._rest_poll_stop.clear()
        if reason:
            self._logger.info(
                "mdm_internal_rest_poller_disabled",
                extra={"event": "mdm_internal_rest_poller_disabled", "reason": reason},
            )

    def set_polling_fallback_streamer(self, streamer: Any) -> None:
        """Attach polling fallback owner. Args: streamer; Returns: none; Raises: none."""
        self._polling_fallback_streamer = streamer

    def request_fallback_refresh(self, symbol: str, *, reason: str) -> bool:
        """Request a fallback refresh. Args: symbol/reason; Returns: dispatched flag; Raises: none."""
        resolved = self._resolve_instrument(symbol)
        streamer = self._polling_fallback_streamer
        owner = "polling_streamer" if streamer is not None else "mdm_rest"
        self._logger.info(
            "MDM_FALLBACK_REFRESH_REQUESTED symbol=%s owner=%s reason=%s",
            resolved.canonical_symbol,
            owner,
            reason,
        )
        if streamer is not None:
            token = resolved.instrument_token
            if token is None:
                self._logger.info(
                    "MDM_FALLBACK_REFRESH_SKIPPED symbol=%s reason=missing_token",
                    resolved.canonical_symbol,
                )
                return False
            try:
                if hasattr(streamer, "subscribe"):
                    streamer.subscribe([int(token)])
                if hasattr(streamer, "start") and hasattr(streamer, "is_running"):
                    if not bool(streamer.is_running()):
                        streamer.start()
                return True
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "MDM_FALLBACK_REFRESH_SKIPPED symbol=%s reason=%s",
                    resolved.canonical_symbol,
                    exc,
                )
                return False
        if not self._should_refresh_symbol(resolved.canonical_symbol):
            self._logger.info(
                "MDM_FALLBACK_REFRESH_SKIPPED symbol=%s reason=rate_limited",
                resolved.canonical_symbol,
            )
            return False
        self._schedule_rest_refresh(resolved.canonical_symbol)
        return True

    def subscribe(self, symbol: str, callback: TickCallback) -> None:
        """Subscribe *callback* to receive normalized ticks for *symbol*."""

        symbol = self._canonical_symbol(symbol)
        with self._lock:
            subscribers = self._subscribers[symbol]
            subscribers.add(callback)
            latest = self._latest_ticks.get(symbol)
            cached_token = self._token_by_symbol.get(symbol)
            self._active_subscribed_symbols.add(symbol)

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

    def unsubscribe(self, symbol: str, callback: TickCallback) -> None:
        """Remove *callback* from subscribers of *symbol*."""

        symbol = self._canonical_symbol(symbol)
        should_unsubscribe = False
        with self._lock:
            callbacks = self._subscribers.get(symbol)
            if callbacks is None:
                return
            callbacks.discard(callback)
            if not callbacks:
                self._subscribers.pop(symbol, None)
                self._active_subscribed_symbols.discard(symbol)
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
            if str(symbol).strip().isdigit():
                with self._lock:
                    resolved_symbol = self._symbol_by_token.get(int(str(symbol).strip()))
                if not resolved_symbol:
                    return None
            else:
                resolved_symbol = self._canonical_symbol(symbol)

        with self._lock:
            tick = self._tick_cache.get(resolved_symbol)
            return dict(tick) if tick is not None else None

    def get_last_tick(self, symbol: str | int) -> dict[str, Any] | None:
        """Return the most recent cached tick for *symbol*.

        Alias for :meth:`get_latest_tick` — provided so callers that use the
        ``get_last_tick`` convention work without modification.

        Args:
            symbol: Instrument identifier (symbol string or integer token).

        Returns:
            dict snapshot of the latest tick, or ``None`` if not yet received.

        Raises:
            None.
        """
        return self.get_latest_tick(symbol)

    async def ensure_fresh_tick(self, symbol: str) -> dict[str, Any] | None:
        """Args: symbol; Returns: cached tick; Raises: none."""

        normalized_symbol = self._canonical_symbol(symbol)
        tick = self.get_latest_tick(normalized_symbol)
        stale_threshold = self._ltp_stale_threshold_for_symbol(normalized_symbol)
        tick_age = self.time_since_last_tick(normalized_symbol)
        tick_stale = tick is None or (
            tick_age is not None
            and stale_threshold > 0.0
            and tick_age > stale_threshold
        )
        if tick_stale:
            self.request_fallback_refresh(normalized_symbol, reason="stale_tick")
        return tick

    def time_since_last_tick(self, symbol: str) -> float | None:
        """Args: symbol; Returns: seconds since last tick; Raises: none."""

        normalized_symbol = self._canonical_symbol(symbol)
        with self._lock:
            wallclock = self._last_tick_time.get(normalized_symbol)
            if wallclock is None:
                tick = self._tick_cache.get(normalized_symbol)
                if tick is None:
                    return None
                wallclock = self._tick_wallclock(tick)
        if not wallclock:
            return None
        return max(time.time() - float(wallclock), 0.0)

    async def _rest_refresh(self, symbol: str) -> None:
        """Args: symbol; Returns: none; Raises: none."""
        resolved = self._resolve_instrument(symbol)
        canonical = resolved.canonical_symbol
        broker_symbol = resolved.broker_symbol
        broker = self._broker or getattr(self, "_kite", None) or getattr(
            self, "_client", None
        )
        if broker is None:
            self._logger.debug(
                "MDM_REST_REFRESH_SKIPPED symbol=%s reason=no_broker", canonical
            )
            return None
        try:
            quote_payload: Any = None
            if canonical.endswith(("CE", "PE")):
                quote_payload = self._broker_quote(broker, broker_symbol)
            elif hasattr(broker, "ltp"):
                quote_payload = broker.ltp([broker_symbol])
            elif hasattr(broker, "get_ltp"):
                quote_payload = {broker_symbol: {"last_price": broker.get_ltp(broker_symbol)}}
            elif hasattr(broker, "get_quote"):
                quote_payload = {broker_symbol: broker.get_quote(broker_symbol)}
            else:
                self._logger.debug(
                    "MDM_REST_REFRESH_SKIPPED symbol=%s reason=no_ltp_method",
                    canonical,
                )
                return None

            quote = quote_payload
            if isinstance(quote_payload, Mapping):
                quote = (
                    quote_payload.get(broker_symbol)
                    or quote_payload.get(canonical)
                    or quote_payload.get(f"NSE:{canonical.split(':')[-1]}")
                )
            if not isinstance(quote, Mapping):
                self._logger.info("MDM_REST_QUOTE_FALLBACK_SKIPPED symbol=%s reason=invalid_quote_payload", canonical)
                return None
            ltp = _coerce_positive_float(quote.get("ltp") or quote.get("last_price"))
            if ltp is None:
                self._logger.info("MDM_REST_QUOTE_FALLBACK_SKIPPED symbol=%s reason=invalid_ltp", canonical)
                return None
            tick = {
                "symbol": canonical,
                "ltp": float(ltp),
                "last_price": float(ltp),
                "bid": _coerce_positive_float(quote.get("bid") or quote.get("buy_price")),
                "ask": _coerce_positive_float(quote.get("ask") or quote.get("sell_price")),
                "received_at": time.time(),
                "timestamp": time.time(),
                "source": "rest_quote" if canonical.endswith(("CE", "PE")) else "rest_ltp",
            }
            with self._lock:
                previous = self._latest_ticks.get(canonical)
            normalized = self._normalize_tick(canonical, tick, previous)
            if normalized is None:
                self._logger.info("MDM_REST_QUOTE_FALLBACK_SKIPPED symbol=%s reason=normalization_failed", canonical)
                return None
            self._emit_tick(canonical, normalized, source="rest")
            if canonical.endswith(("CE", "PE")):
                self._logger.info(
                    "MDM_REST_QUOTE_FALLBACK_USED symbol=%s bid=%s ask=%s",
                    canonical,
                    tick.get("bid"),
                    tick.get("ask"),
                )
            else:
                self._logger.info(
                    "MDM_REST_FALLBACK_USED symbol=%s ltp=%s",
                    canonical,
                    float(ltp),
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "MDM_REST_REFRESH_FAILED symbol=%s error=%s", canonical, exc
            )
        finally:
            with suppress(KeyError):
                self._rest_refresh_inflight.remove(canonical)
        return None

    def _schedule_rest_refresh(self, symbol: str) -> None:
        """Args: symbol; Returns: none; Raises: none."""

        canonical = self._canonical_symbol(symbol)
        if canonical in self._rest_refresh_inflight:
            return
        self._rest_refresh_inflight.add(canonical)
        try:
            asyncio.get_running_loop()
            safe_task(self._rest_refresh(canonical))
        except RuntimeError:
            thread = threading.Thread(
                target=lambda: asyncio.run(self._rest_refresh(canonical)),
                name=f"mdm-rest-refresh-{canonical}",
                daemon=True,
            )
            thread.start()

    def _should_refresh_symbol(self, symbol: str, interval_sec: float = 5.0) -> bool:
        """Rate-limit REST refresh attempts per symbol."""
        canonical = self._canonical_symbol(symbol)
        now = time.monotonic()
        with self._lock:
            last = self._last_rest_refresh_attempt.get(canonical, 0.0)
            if now - last < max(interval_sec, 0.0):
                return False
            self._last_rest_refresh_attempt[canonical] = now
        return True

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

    def get_cached_ltp(
        self,
        symbol: str,
        *,
        max_age_seconds: float | None = None,
        require_ws: bool = False,
    ) -> float | None:
        """Cached-only LTP accessor. Args: symbol/guards. Returns: ltp. Raises: none."""
        canonical_symbol = self._canonical_symbol(symbol)
        tick = self.get_latest_tick(canonical_symbol)
        if not isinstance(tick, Mapping):
            return None
        if max_age_seconds is not None:
            age = self.time_since_last_tick(canonical_symbol)
            if age is None or age > float(max_age_seconds):
                return None
        source = str(
            tick.get("source") or self._last_tick_source.get(canonical_symbol) or ""
        ).lower()
        if require_ws and source not in {"ws", "ws_full", "full"}:
            return None
        price = _coerce_positive_float(
            tick.get("last_price")
            or tick.get("ltp")
            or tick.get("price")
            or tick.get("close")
        )
        return float(price) if price and price > 0 else None

    def get_ltp(
        self,
        symbol: str,
        *,
        allow_rest_fallback: bool | None = None,
    ) -> float | None:
        """Return latest traded price for symbol with 2-second stale guard.

        If the cached tick is older than 2 seconds, falls back to broker REST
        quote to guarantee a fresh price is always available.

        Args:
            symbol: Trading symbol identifier.

        Returns:
            float | None: Latest price or None if unavailable from all sources.
        """
        canonical_symbol = self._canonical_symbol(symbol)
        stale_seconds = self._ltp_stale_threshold_for_symbol(canonical_symbol)
        now = time.time()
        cached = self.get_cached_ltp(
            canonical_symbol,
            max_age_seconds=stale_seconds,
            require_ws=False,
        )
        if cached is not None:
            return cached

        # 1. Try tick cache with freshness check
        tick = self.get_latest_tick(canonical_symbol)
        tick_age: float | None = None
        if tick is not None:
            last_ts = self._last_tick_time.get(canonical_symbol)
            if last_ts is not None:
                tick_age = now - last_ts
            try:
                ltp = float(tick["ltp"])
                if ltp > 0:
                    if tick_age is None or tick_age <= stale_seconds:
                        return ltp
                    # Tick exists but is stale — log and fall through to broker
                    last_warn = self._ltp_stale_warn_last.get(canonical_symbol, 0.0)
                    market_state = get_market_state()
                    log_interval = 900.0 if market_state != MarketState.OPEN else 60.0
                    if now - last_warn >= log_interval:
                        self._ltp_stale_warn_last[canonical_symbol] = now
                        if market_state == MarketState.OPEN:
                            self._logger.warning(
                                "STALE_DATA symbol=%s age=%.2fs threshold=%.2fs — falling back to broker REST",
                                canonical_symbol,
                                tick_age,
                                stale_seconds,
                                extra={
                                    "event": "STALE_DATA",
                                    "symbol": canonical_symbol,
                                    "age_s": round(float(tick_age), 3),
                                    "threshold_s": stale_seconds,
                                },
                            )
                        else:
                            self._logger.debug(
                                "OFF_MARKET_LTP_STALE symbol=%s age=%.2fs threshold=%.2fs state=%s",
                                canonical_symbol,
                                tick_age,
                                stale_seconds,
                                market_state.value,
                                extra={
                                    "event": "OFF_MARKET_LTP_STALE",
                                    "symbol": canonical_symbol,
                                    "age_s": round(float(tick_age), 3),
                                    "threshold_s": stale_seconds,
                                    "state": market_state.value,
                                },
                            )
            except (KeyError, TypeError, ValueError):
                pass

        # 2. Broker REST fallback
        policy = getattr(self, "_md_policy", MarketDataPolicy.from_env())
        ws_started = bool(
            getattr(self, "_ws_started", False)
            or getattr(self, "_websocket_started", False)
            or getattr(self, "ws_connected", False)
        )
        fallback_active = bool(
            getattr(self, "_polling_fallback_active", False)
            or getattr(self, "_rest_fallback_active", False)
            or getattr(self, "_fallback_active", False)
        )
        if policy.should_suppress_rest_quote(
            symbol=canonical_symbol,
            ws_started=ws_started,
            fallback_active=fallback_active,
            explicit_allow_rest=allow_rest_fallback,
        ):
            self._logger.debug(
                "MDM_REST_QUOTE_SUPPRESSED symbol=%s ws_started=%s fallback_active=%s allow_rest_fallback=%s",
                canonical_symbol,
                ws_started,
                fallback_active,
                allow_rest_fallback,
                extra={
                    "event": "MDM_REST_QUOTE_SUPPRESSED",
                    "symbol": canonical_symbol,
                    "ws_started": ws_started,
                    "fallback_active": fallback_active,
                    "allow_rest_fallback": allow_rest_fallback,
                },
            )
            return None

        broker = getattr(self, "_broker", None)
        if broker:
            try:
                t_before_rest = time.time()
                broker_symbol = policy.to_broker_quote_symbol(canonical_symbol)
                quote = broker.get_quote(broker_symbol)
                if isinstance(quote, dict):
                    price = quote.get("last_price") or quote.get("ltp")
                    if price:
                        p = float(price)
                        if p > 0:
                            self._logger.info(
                                "MDM_REST_FALLBACK_USED symbol=%s reason=stale_ws_tick price=%.2f",
                                canonical_symbol,
                                p,
                                extra={"event": "MDM_REST_FALLBACK_USED", "symbol": canonical_symbol, "reason": "stale_ws_tick", "price": p},
                            )
                            # Repair symbol-level cache so the next call to
                            # get_ltp does not hit the stale guard again.
                            # Skip if a live tick arrived while the REST call
                            # was in flight — that tick is already fresher.
                            try:
                                with self._lock:
                                    live_tick_arrived = (
                                        self._last_tick_time.get(canonical_symbol, 0)
                                        > t_before_rest
                                    )
                                if not live_tick_arrived:
                                    prepared = self._prepare_rest_tick(quote, source="rest")
                                    with self._lock:
                                        prev = self._latest_ticks.get(canonical_symbol)
                                    normalized_repair = self._normalize_tick(
                                        canonical_symbol, prepared, prev
                                    )
                                    if normalized_repair is not None:
                                        self._store_tick(canonical_symbol, normalized_repair)
                                        try:
                                            self.update_authoritative_ticks([normalized_repair])
                                        except Exception:
                                            pass
                                    else:
                                        _now_ts = time.time()
                                        _now_ms_v = self._now_ms()
                                        with self._lock:
                                            self._latest_ticks[canonical_symbol] = {
                                                "symbol": canonical_symbol,
                                                "ltp": p,
                                                "price": p,
                                                "timestamp": _now_ts,
                                                "source": "rest",
                                            }
                                            self._last_tick_time[canonical_symbol] = _now_ts
                                            self._last_tick_wallclock[canonical_symbol] = _now_ts
                                            self._last_quote_ts_ms[canonical_symbol] = _now_ms_v
                                            self._last_tick_source[canonical_symbol] = "rest"
                                        try:
                                            self.update_authoritative_ticks(
                                                [
                                                    {
                                                        "symbol": canonical_symbol,
                                                        "ltp": p,
                                                        "price": p,
                                                        "timestamp": _now_ts,
                                                        "source": "rest",
                                                    }
                                                ]
                                            )
                                        except Exception:
                                            pass
                            except Exception:  # noqa: BLE001
                                pass
                            return p
            except Exception as exc:
                self._logger.warning(
                    "broker.get_quote failed for %s: %s", canonical_symbol, exc
                )

        self._logger.warning("get_ltp: no price available for %s", canonical_symbol)
        return None

    def get_latest_price(
        self,
        symbol: str,
        *,
        allow_rest_fallback: bool | None = None,
    ) -> float | None:
        """Alias for get_ltp — preserves backward compatibility."""
        return self.get_ltp(symbol, allow_rest_fallback=allow_rest_fallback)

    def get_fresh_spot_tick(
        self,
        symbol: str = "NSE:NIFTY",
        *,
        max_age_seconds: float | None = None,
        require_ws: bool = True,
    ) -> dict[str, Any] | None:
        """Return the cached spot tick if it satisfies freshness/source guards.

        In live startup we must avoid building the option universe from a stale
        REST quote.  This helper enforces a tick age window and (optionally) a
        WebSocket-source requirement so callers can prove fresh exchange data
        before arming live trading.

        Args:
            symbol: Spot symbol to inspect (default ``NSE:NIFTY``).
            max_age_seconds: Maximum acceptable tick age in seconds.  When
                ``None`` the per-symbol stale threshold is used.
            require_ws: When ``True``, only WebSocket-sourced ticks are
                accepted; REST/polling ticks are rejected.

        Returns:
            A copy of the latest tick dict that passes all guards, or ``None``
            when no fresh proof is available.
        """

        canonical = self._canonical_symbol(symbol)
        tick = self.get_latest_tick(canonical)
        if not isinstance(tick, Mapping):
            return None

        age = self.time_since_last_tick(canonical)
        threshold = (
            float(max_age_seconds)
            if max_age_seconds is not None
            else float(self._ltp_stale_threshold_for_symbol(canonical) or 120.0)
        )
        if age is None or threshold <= 0 or age > threshold:
            return None

        source = str(
            tick.get("source") or self._last_tick_source.get(canonical) or ""
        ).lower()
        if require_ws and source not in {"ws", "ws_full", "full"}:
            return None

        price = _coerce_positive_float(
            tick.get("last_price")
            or tick.get("ltp")
            or tick.get("price")
            or tick.get("close")
        )
        if price is None or price <= 0:
            return None

        return dict(tick)

    async def wait_for_fresh_spot_tick(
        self,
        symbol: str = "NSE:NIFTY",
        *,
        timeout: float = 15.0,
        max_age_seconds: float | None = None,
        require_ws: bool = True,
        poll_interval: float = 0.25,
    ) -> dict[str, Any] | None:
        """Block until ``symbol`` has a fresh spot tick or ``timeout`` elapses.

        Args:
            symbol: Spot symbol to wait on.
            timeout: Maximum wait in seconds.
            max_age_seconds: Override per-symbol stale threshold.
            require_ws: Require a WebSocket-sourced tick when ``True``.
            poll_interval: Seconds between checks while waiting.

        Returns:
            The fresh tick dict, or ``None`` if no fresh tick arrived in time.
        """

        deadline = time.monotonic() + max(0.0, float(timeout))
        interval = max(0.05, float(poll_interval))
        while True:
            tick = self.get_fresh_spot_tick(
                symbol,
                max_age_seconds=max_age_seconds,
                require_ws=require_ws,
            )
            if tick is not None:
                return tick
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(interval)

    def _resolve_underlying_price(self, symbol: str) -> float | None:
        canonical_symbol = self._canonical_symbol(symbol)
        tick_price = self.get_latest_price(canonical_symbol)
        if tick_price is not None:
            return tick_price
        broker = getattr(self, "_broker", None)
        if broker is None or not hasattr(broker, "get_quote"):
            return None
        try:
            quote = broker.get_quote(canonical_symbol)
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

        canonical_symbol = self._canonical_symbol(symbol)
        candidates: list[str | int] = self._candidate_quote_keys(canonical_symbol)
        if not candidates:
            candidates = [canonical_symbol]
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
                raw_quote = self._broker.get_quote(canonical_symbol)
            except Exception as exc:  # noqa: BLE001
                if self._is_quote_access_denied(exc):
                    self._mark_quote_api_status(available=False, error="access_denied")
                error_text = str(exc)
                is_recoverable_miss = "Quote data missing" in error_text
                log_fn = self._logger.warning if is_recoverable_miss else self._logger.error
                log_fn(
                    "Failure in pull_quote direct get_quote: %s",
                    exc,
                    extra={
                        "event": (
                            "mdm_pull_quote_direct_miss"
                            if is_recoverable_miss
                            else "mdm_pull_quote_direct_error"
                        ),
                        "symbol": canonical_symbol,
                        "error": error_text,
                    },
                    exc_info=None if is_recoverable_miss else exc,
                )
                raw_quote = None
            if isinstance(raw_quote, Mapping):
                self._mark_quote_api_status(available=True)
                quote = dict(raw_quote)
        if quote is None:
            with self._lock:
                cached_tick = self._latest_ticks.get(canonical_symbol)
            if isinstance(cached_tick, Mapping) and cached_tick:
                cached_quote = dict(cached_tick)
                cached_quote.setdefault("source", "cache")
                return cached_quote
            if canonical_symbol == "NSE:NIFTY":
                self._logger.warning(
                    "nifty_quote_pull_failed",
                    extra={
                        "event": "nifty_quote_pull_failed",
                        "symbol": canonical_symbol,
                        "reason": "no_quote_from_broker_or_cache",
                    },
                )
            return {"symbol": canonical_symbol}
        quote = self._prepare_rest_tick(quote, source="rest")
        with self._lock:
            previous = self._latest_ticks.get(canonical_symbol)

        normalized = self._normalize_tick(canonical_symbol, quote, previous)
        if normalized is not None:
            self._store_tick(canonical_symbol, normalized)
            return normalized
        return {"symbol": canonical_symbol, **quote}

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
            if bar_count >= self._min_required_bars:
                self._hydration_status[normalized] = "READY"
            else:
                self._hydration_status[normalized] = "HYDRATING"
                now = time.monotonic()
                last_log_ts = float(self._hydration_log_ts.get(normalized, 0.0))
                should_log = (now - last_log_ts) >= 15.0
                if should_log:
                    self._hydration_log_ts[normalized] = now
                    _reqs = self._readiness_requirements
                    _core: set[str] = {
                        v
                        for k in ("spot", "futures", "atm_ce", "atm_pe")
                        if (v := _reqs.get(k))
                    }
                    is_core = normalized in _core
                    if (self.hydration_complete or self.ready) and is_core:
                        self._logger.warning(
                            "insufficient_bars_for_strategy",
                            extra={
                                "event": "insufficient_bars_for_strategy",
                                "symbol": normalized,
                                "bars": bar_count,
                                "required": self._min_required_bars,
                            },
                        )
                    else:
                        self._logger.debug(
                            "symbol_hydration_in_progress",
                            extra={
                                "event": "symbol_hydration_in_progress",
                                "symbol": normalized,
                                "bars": bar_count,
                                "required": self._min_required_bars,
                            },
                        )
            if self._hydration_status.get(normalized) == "READY":
                self._logger.info(
                    "Condition met: mdm_hydration_progress",
                    extra={
                        "event": "mdm_hydration_progress",
                        "symbol": normalized,
                        "bars": bar_count,
                        "required": self._min_required_bars,
                        "status": "READY",
                    },
                )
        except Exception as exc:
            self._logger.error(
                "Failure in update_hydration_status: %s", exc, exc_info=exc
            )

    def is_symbol_ready(self, symbol: str) -> bool:
        """Args: symbol; Returns: bool; Raises: none."""
        normalized = normalize_symbol(str(symbol or ""))
        with self._lock:
            return len(self._history.get(normalized, ())) >= self._min_required_bars

    def is_ready(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""
        with self._lock:
            active = sorted(self._active_subscribed_symbols)
            if not active:
                return False
            return all(
                len(self._history.get(symbol, ())) >= self._min_required_bars
                for symbol in active
            )

    async def wait_until_ready(self, timeout: float = 30.0) -> None:
        """Args: timeout; Returns: None; Raises: None."""

        deadline = time.monotonic() + max(timeout, 0.0)
        self.ready = False
        self.degraded = False
        min_bars = self._min_required_bars
        bars: dict[str, int] = {}
        while time.monotonic() <= deadline:
            with self._lock:
                subscribed_symbols = sorted(self._active_subscribed_symbols)
                min_bars = self._min_required_bars
                bars = {
                    symbol: max(
                        len(self._history.get(symbol, ())),
                        len(self._ohlc.get(self._bar_symbol_key(symbol), ())),
                    )
                    for symbol in subscribed_symbols
                }
            requirements = dict(self._readiness_requirements)
            readiness_state = self._readiness_state(bars, min_bars, requirements)
            self._last_readiness_state = dict(readiness_state)
            valid_symbols = [s for s, count in bars.items() if count >= min_bars]
            spot_symbol = str(requirements.get("spot") or "NSE:NIFTY")
            spot_age_ms = self.symbol_data_age_ms_or_none(spot_symbol)
            threshold_ms = 120000
            spot_fresh = spot_age_ms is not None and 0 <= int(spot_age_ms) <= threshold_ms
            if readiness_state["spot_ready"] and spot_fresh and not self._spot_ready_logged:
                spot_source = str(self._last_tick_source.get(spot_symbol, "unknown"))
                source = "ws" if spot_source == "ws" else "rest_ltp"
                token = self._token_by_symbol.get(spot_symbol)
                self._spot_ready_logged = True
                self._logger.info(
                    "SPOT_READY symbol=%s source=%s age_ms=%s token=%s",
                    spot_symbol,
                    source,
                    max(0, int(spot_age_ms)),
                    token,
                    extra={
                        "event": "SPOT_READY",
                        "symbol": spot_symbol,
                        "source": source,
                        "age_ms": max(0, int(spot_age_ms)),
                        "token": token,
                    },
                )
            elif not (readiness_state["spot_ready"] and spot_fresh):
                reason = "no_tick_received" if spot_age_ms is None else "stale_tick"
                log_throttled(
                    self._logger,
                    "spot_not_ready_state",
                    "SPOT_NOT_READY symbol=%s reason=%s age_ms=%s threshold_ms=%s"
                    % (spot_symbol, reason, spot_age_ms, threshold_ms),
                    interval_sec=30.0,
                    level=logging.INFO,
                    extra={
                        "event": "SPOT_NOT_READY",
                        "symbol": spot_symbol,
                        "reason": reason,
                        "age_ms": spot_age_ms,
                        "threshold_ms": threshold_ms,
                    },
                )
            if readiness_state["hard_ready"]:
                self._logger.info(
                    "DATA_PIPELINE_READY hard_ready=%s spot_ready=%s symbols_ready=%s",
                    True,
                    readiness_state["spot_ready"],
                    len(valid_symbols),
                    extra={
                        "event": "DATA_PIPELINE_READY",
                        "hard_ready": True,
                        "spot_ready": readiness_state["spot_ready"],
                        "symbols_ready": len(valid_symbols),
                    },
                )
                self.ready = True
                self.degraded = False
                self.hydration_complete = True
                self._logger.info(
                    "Condition met: mdm_ready",
                    extra={
                        "event": "mdm_ready",
                        "symbols": valid_symbols,
                        "requirements": requirements,
                        "spot_ready": readiness_state["spot_ready"],
                    },
                )
                return
            log_throttled(
                self._logger,
                "data_pipeline_not_ready",
                "DATA_PIPELINE_NOT_READY hard_ready=%s spot_ready=%s missing=%s"
                % (
                    False,
                    readiness_state["spot_ready"],
                    readiness_state.get("missing_hard") or [],
                ),
                interval_sec=30,
                level=logging.INFO,
                extra={
                    "event": "DATA_PIPELINE_NOT_READY",
                    "hard_ready": False,
                    "spot_ready": readiness_state["spot_ready"],
                    "missing": readiness_state.get("missing_hard") or [],
                },
            )
            await asyncio.sleep(0.1)

        market_state = get_market_state()
        if market_state != MarketState.OPEN:
            self._logger.info(
                "MDM_READY_BYPASS_OFF_MARKET state=%s dashboard_ready=true trading_ready=false",
                market_state.value,
                extra={
                    "event": "MDM_READY_BYPASS_OFF_MARKET",
                    "state": market_state.value,
                    "dashboard_ready": True,
                    "trading_ready": False,
                    "off_market_ready_bypass": True,
                },
            )
            self.ready = True
            self.degraded = False
            self._last_readiness_state.update(
                {
                    "off_market_ready_bypass": True,
                    "dashboard_ready": True,
                    "trading_ready": False,
                }
            )
            return

        if self.hydration_complete:
            self._logger.warning(
                "Entering DEGRADED mode due to insufficient live data "
                "(timeout=%.1fs, min_bars=%d, bars=%s, hydration_complete=%s)",
                timeout,
                min_bars,
                bars,
                self.hydration_complete,
            )
            self.ready = False
            self.degraded = True
        else:
            timed_out_state = self._readiness_state(
                bars, min_bars, self._readiness_requirements
            )
            self._last_readiness_state = dict(timed_out_state)
            self._logger.warning(
                "Readiness timeout before hydration completed "
                "(timeout=%.1fs, min_bars=%d, bars=%s, requirements=%s, "
                "hard_ready=%s, spot_ready=%s, missing_hard=%s)",
                timeout,
                min_bars,
                bars,
                self._readiness_requirements,
                timed_out_state["hard_ready"],
                timed_out_state["spot_ready"],
                timed_out_state["missing_hard"],
            )
            self.ready = False
            self.degraded = False
        return

    def _readiness_passes(
        self,
        bars: Mapping[str, int],
        min_bars: int,
        requirements: Mapping[str, Any],
    ) -> bool:
        """Evaluate strict readiness rules for startup/live gating."""
        return bool(self._readiness_state(bars, min_bars, requirements)["hard_ready"])

    def _readiness_state(
        self,
        bars: Mapping[str, int],
        min_bars: int,
        requirements: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Classify readiness into hard-trading and soft-spot states."""
        if not requirements:
            any_ready = any(count >= min_bars for count in bars.values())
            spot_fresh = False
            try:
                spot_fresh = bool(
                    self._is_symbol_fresh("NSE:NIFTY", self._tick_stale_threshold_ms)
                )
            except Exception:
                spot_fresh = False
            missing: list[str] = []
            if not any_ready:
                missing.append("readiness_requirements_missing")
            if not spot_fresh:
                missing.append("fresh_spot_tick_missing")
            return {
                "hard_ready": bool(any_ready),
                "spot_ready": bool(spot_fresh),
                "missing_hard": missing,
            }
        spot = str(requirements.get("spot") or "")
        futures = str(requirements.get("futures") or "")
        atm_ce = str(requirements.get("atm_ce") or "")
        atm_pe = str(requirements.get("atm_pe") or "")
        options = [str(s) for s in requirements.get("options") or []]

        missing_hard: list[str] = []
        if futures and bars.get(futures, 0) < min_bars:
            missing_hard.append("futures")
        if atm_ce and bars.get(atm_ce, 0) < min_bars:
            missing_hard.append("atm_ce")
        if atm_pe and bars.get(atm_pe, 0) < min_bars:
            missing_hard.append("atm_pe")

        if not atm_ce or not atm_pe:
            ce_ready = any(
                sym.endswith("CE") and bars.get(sym, 0) >= min_bars for sym in options
            )
            pe_ready = any(
                sym.endswith("PE") and bars.get(sym, 0) >= min_bars for sym in options
            )
            if not ce_ready:
                missing_hard.append("options_ce")
            if not pe_ready:
                missing_hard.append("options_pe")

        spot_ready = True
        if spot:
            try:
                spot_ready = bool(self._is_symbol_fresh(spot, self._tick_stale_threshold_ms))
            except Exception:
                spot_ready = False
        return {
            "hard_ready": len(missing_hard) == 0,
            "spot_ready": spot_ready,
            "missing_hard": missing_hard,
        }

    def readiness_state_snapshot(self) -> dict[str, Any]:
        """Return latest readiness classification snapshot."""
        return dict(self._last_readiness_state)

    def readiness_snapshot(self) -> dict[str, Any]:
        """Backward-compatible alias for readiness_state_snapshot."""
        return self.readiness_state_snapshot()

    def hard_ready(self) -> bool:
        """Args: none; Returns: hard readiness flag; Raises: none."""
        return bool(self._last_readiness_state.get("hard_ready"))

    def spot_ready(self) -> bool:
        """Args: none; Returns: spot readiness flag; Raises: none."""
        return bool(self._last_readiness_state.get("spot_ready"))

    def missing_hard(self) -> list[str]:
        """Args: none; Returns: missing hard dependencies; Raises: none."""
        return list(self._last_readiness_state.get("missing_hard") or [])

    def quote_api_available(self) -> bool:
        """Return broker quote API capability status."""
        return bool(self._quote_api_available)

    def quote_api_status_snapshot(self) -> dict[str, Any]:
        """Return quote API capability details."""
        return {
            "available": bool(self._quote_api_available),
            "error": self._quote_api_error,
            "last_checked_at": self._quote_api_last_checked_at,
        }

    def has_ws_tradable_quote(
        self, symbols: Sequence[str] | None = None
    ) -> bool:
        """Return whether websocket/depth tradable quote proof exists."""
        with self._lock:
            candidate_symbols = (
                [self._canonical_symbol(sym) for sym in symbols]
                if symbols is not None
                else list(self._active_subscribed_symbols)
            )
            for symbol in candidate_symbols:
                tick = self._latest_ticks.get(symbol)
                if not isinstance(tick, Mapping):
                    continue
                source = str(
                    tick.get("source")
                    or self._last_tick_source.get(symbol)
                    or ""
                ).lower()
                bid_ask_source = str(tick.get("bid_ask_source") or "").lower()
                if bool(tick.get("tradable_quote")) and (
                    source in {"ws", "ws_full", "full"}
                    or bid_ask_source in {"market_depth", "ws_full"}
                ):
                    return True
        return False

    def has_fresh_ws_ltp(
        self, symbols: Sequence[str] | None = None, max_age_seconds: float = 5.0
    ) -> bool:
        """Return whether fresh websocket LTP proof exists for active symbols."""
        now = time.time()
        max_age = max(float(max_age_seconds), 0.1)
        allowed_sources = {"ws", "ws_full", "full"}
        with self._lock:
            candidate_symbols = (
                [self._canonical_symbol(sym) for sym in symbols]
                if symbols is not None
                else list(self._active_subscribed_symbols)
            )
            if not candidate_symbols:
                candidate_symbols = list(self._latest_ticks.keys())
            for symbol in candidate_symbols:
                tick = self._latest_ticks.get(symbol)
                if not isinstance(tick, Mapping):
                    continue
                source = str(
                    tick.get("source")
                    or self._last_tick_source.get(symbol)
                    or ""
                ).lower()
                if source not in allowed_sources:
                    continue
                price = tick.get("ltp", tick.get("last_price", tick.get("price", 0)))
                try:
                    ltp = float(price)
                except (TypeError, ValueError):
                    continue
                if ltp <= 0:
                    continue
                tick_ts = tick.get("exchange_timestamp") or tick.get("timestamp")
                age_seconds: float | None = None
                if isinstance(tick_ts, datetime):
                    ts = tick_ts if tick_ts.tzinfo is not None else tick_ts.replace(tzinfo=timezone.utc)
                    age_seconds = max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
                elif tick_ts is not None:
                    try:
                        age_seconds = max(now - float(tick_ts), 0.0)
                    except (TypeError, ValueError):
                        age_seconds = None
                if age_seconds is None or age_seconds <= max_age:
                    return True
        return False

    def _mark_quote_api_status(
        self, *, available: bool, error: str | None = None
    ) -> None:
        """Update quote API status. Args: available/error. Returns: None. Raises: none."""
        self._quote_api_available = bool(available)
        self._quote_api_error = str(error) if error else None
        self._quote_api_last_checked_at = time.time()

    def _is_quote_access_denied(self, exc: Exception) -> bool:
        """Detect terminal quote access-denied errors."""
        text = str(exc).lower()
        return (
            "access denied" in text
            or "quote api denied" in text
            or "http 403" in text
            or "403" in text
            or "forbidden" in text
        )

    def _is_symbol_fresh(self, symbol: str, max_age_ms: int) -> bool:
        """Return whether a symbol has a fresh enough tick age."""
        return self.symbol_data_age_ms(symbol) <= int(max(max_age_ms, 1))

    def _active_option_symbols(self) -> list[str]:
        """Return currently active option symbols used for hard feed checks."""
        requirements = dict(self._readiness_requirements)
        symbols = [str(s) for s in requirements.get("options") or [] if s]
        if not symbols:
            with self._lock:
                symbols = [
                    symbol
                    for symbol in self._active_subscribed_symbols
                    if symbol.endswith("CE") or symbol.endswith("PE")
                ]
        return sorted(set(symbols))

    def _is_spot_fresh(self, max_age_ms: int | None = None) -> bool:
        """Return whether spot is fresh enough for advisory reference."""
        threshold = int(max_age_ms or self._tick_stale_threshold_ms)
        spot_symbol = str(self._readiness_requirements.get("spot") or "NSE:NIFTY")
        return self._is_symbol_fresh(spot_symbol, threshold)

    def _is_futures_fresh(self, max_age_ms: int | None = None) -> bool:
        """Return whether futures feed is fresh enough for trading safety."""
        threshold = int(max_age_ms or self._tick_stale_threshold_ms)
        futures_symbol = str(self._readiness_requirements.get("futures") or "")
        if not futures_symbol:
            return False
        return self._is_symbol_fresh(futures_symbol, threshold)

    def _are_active_options_fresh(self, max_age_ms: int | None = None) -> bool:
        """Return whether active options feed is fresh enough for trading safety."""
        threshold = int(max_age_ms or self._tick_stale_threshold_ms)
        option_symbols = self._active_option_symbols()
        if not option_symbols:
            return False
        return all(self._is_symbol_fresh(symbol, threshold) for symbol in option_symbols)

    def trading_feed_health(self, max_age_ms: int | None = None) -> dict[str, Any]:
        """Return trading-critical feed health where spot is advisory only."""
        threshold = int(max_age_ms or self._tick_stale_threshold_ms)
        requirements = dict(self._readiness_requirements)
        spot_symbol = str(requirements.get("spot") or "NSE:NIFTY")
        futures_symbol = str(requirements.get("futures") or "")
        option_symbols = self._active_option_symbols()
        futures_fresh = self._is_futures_fresh(threshold)
        options_fresh = self._are_active_options_fresh(threshold)
        spot_fresh = self._is_spot_fresh(threshold)
        return {
            "trading_feed_healthy": futures_fresh and options_fresh,
            "futures_fresh": futures_fresh,
            "options_fresh": options_fresh,
            "spot_fresh": spot_fresh,
            "spot_symbol": spot_symbol,
            "spot_age_ms": self.symbol_data_age_ms(spot_symbol),
            "futures_symbol": futures_symbol,
            "futures_age_ms": self.symbol_data_age_ms(futures_symbol)
            if futures_symbol
            else 1_000_000_000,
            "option_symbols": option_symbols,
            "threshold_ms": threshold,
        }

    def ensure_spot_reference_fresh(
        self,
        *,
        symbol: str = "NSE:NIFTY",
        stale_after_ms: int | None = None,
        min_refresh_interval_s: float = 3.0,
    ) -> bool:
        """Soft-check spot and trigger REST refresh without global fallback."""
        canonical_symbol = self._canonical_symbol(symbol)
        threshold = int(stale_after_ms or self._tick_stale_threshold_ms)
        age_ms = self.symbol_data_age_ms(canonical_symbol)
        if age_ms <= threshold:
            return True
        self._logger.warning(
            "spot_quote_stale_soft_warning symbol=%s age_ms=%s threshold_ms=%s",
            canonical_symbol,
            age_ms,
            threshold,
        )
        now_mono = time.monotonic()
        if now_mono - self._spot_refresh_last_attempt_mono < max(
            min_refresh_interval_s, 0.0
        ):
            return False
        self._spot_refresh_last_attempt_mono = now_mono
        self._logger.info("spot_rest_refresh_attempt symbol=%s", canonical_symbol)
        self.request_fallback_refresh(canonical_symbol, reason="spot_reference_stale")
        return False

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
            return False
        if not self._ws.is_connected():
            return False
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
        """Wire asyncio loop and ensure the tick consumer is running."""
        try:
            if self._main_loop is not None and self._main_loop is not loop:
                self._logger.warning("Event loop already wired — ignoring rewire")
                return

            self._main_loop = loop
            self._ensure_tick_consumer(reason="event_loop_wired")
        except Exception as e:
            self._logger.error(
                "Failure in MarketDataManager.set_event_loop: %s",
                e,
                exc_info=True,
            )

    def _ensure_tick_consumer(self, reason: str) -> None:
        """Start the async tick consumer exactly once on the owning loop thread."""
        loop = self._main_loop
        if loop is None:
            return

        def _start() -> None:
            task = getattr(self, "_tick_consumer_task", None)
            if task is None or task.done():
                self._tick_consumer_task = loop.create_task(self._consume_ticks())
                self._logger.info(
                    "MDM_TICK_CONSUMER_STARTED reason=%s",
                    reason,
                    extra={
                        "event": "MDM_TICK_CONSUMER_STARTED",
                        "reason": reason,
                    },
                )

        if not loop.is_running():
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            _start()
        else:
            loop.call_soon_threadsafe(_start)

    async def push_tick(self, raw_tick: dict[str, Any]) -> None:
        """Queue one raw websocket tick for deterministic consumption."""
        await self._tick_queue.put(dict(raw_tick))

    def _enqueue_tick_threadsafe(self, tick: dict[str, Any]) -> None:
        """Enqueue websocket ticks from sync callbacks onto the async queue.

        This function is called from the KiteTicker WS thread and MUST be
        non-blocking.  We never perform processing work here — we only
        enqueue.  If the queue is full we drop the oldest tick so a stall
        in the consumer cannot back-pressure onto the WS thread.  If no
        asyncio loop is registered we spin up a dedicated worker thread
        that drains the queue — never the WS thread itself.
        """
        try:
            tick_payload = dict(tick)
        except Exception:  # pragma: no cover — defensive
            return
        tick_payload.setdefault("_enqueued_monotonic", time.monotonic())

        loop = self._main_loop
        if loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.push_tick(tick_payload), loop)
            except Exception as exc:  # pragma: no cover — defensive
                self._logger.debug("WS enqueue via loop failed: %s", exc)
            return

        # No event loop available — ensure a worker thread exists so WS
        # thread never performs processing work inline.
        self._ensure_tick_worker()
        try:
            self._tick_queue.put_nowait(tick_payload)
        except asyncio.QueueFull:
            # Drop oldest to make room; throttle-log every 5 s.
            self._tick_queue_dropped += 1
            try:
                _ = self._tick_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._tick_queue.put_nowait(tick_payload)
            except asyncio.QueueFull:  # pragma: no cover
                pass
            now = time.monotonic()
            if now - self._last_tick_queue_full_log > 5.0:
                self._last_tick_queue_full_log = now
                self._logger.warning(
                    "Tick queue full — dropped %d ticks so far",
                    self._tick_queue_dropped,
                )
        except Exception as exc:  # pragma: no cover — defensive
            self._logger.debug("WS enqueue failed: %s", exc)

    def _ensure_tick_worker(self) -> None:
        """Start the fallback drain thread once, on first need."""
        # TODO: fallback worker currently uses asyncio.Queue from a worker thread.
        # Runtime path normally uses run_coroutine_threadsafe() onto _main_loop.
        # If fallback becomes production-critical, replace with queue.Queue for
        # cross-thread safety.
        t = self._tick_worker_thread
        if t is not None and t.is_alive():
            return
        self._tick_worker_stop.clear()
        self._tick_worker_thread = threading.Thread(
            target=self._tick_worker_loop,
            name="mdm-tick-worker",
            daemon=True,
        )
        self._tick_worker_thread.start()

    def _tick_worker_loop(self) -> None:
        """Drain the tick queue off-thread when no asyncio loop is running."""
        while not self._tick_worker_stop.is_set():
            # Prefer the loop when it comes online — self-terminate.
            if self._main_loop is not None and self._main_loop.is_running():
                return
            try:
                raw = self._tick_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Light spin — 25 ms pacing keeps latency low without
                # burning CPU.
                if self._tick_worker_stop.wait(0.025):
                    return
                continue
            try:
                self._process_queued_tick(raw)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug("tick worker drain error: %s", exc)

    def _on_tick(self, tick: dict[str, Any]) -> None:
        """Legacy tick-bus hook routed to queue ingestion."""
        self._enqueue_tick_threadsafe(tick)

    def _is_nifty_spot_tick(self, symbol: str | None = None, token: int | None = None) -> bool:
        """Return whether tick belongs to NIFTY spot aliases/token."""
        s = (symbol or "").upper().replace(" ", "")
        return token == NIFTY_SPOT_TOKEN or s in {"NIFTY", "NSE:NIFTY", "NIFTY50", "NSE:NIFTY50"}

    def _record_ws_arrival_fast(
        self,
        *,
        symbol: str,
        token: int | None,
        ltp: Any,
        raw_tick: dict[str, Any],
    ) -> None:
        """Fast freshness cache update from WS thread without candle building."""
        try:
            canonical = self._canonical_symbol(symbol)
            price = float(ltp)
            if price <= 0:
                return
            now = time.time()
            payload = dict(raw_tick)
            payload["symbol"] = canonical
            payload["ltp"] = price
            payload.setdefault("last_price", price)
            payload["source"] = "ws"
            payload["received_at"] = now
            payload.setdefault("timestamp", now)
            if token is not None:
                payload["instrument_token"] = int(token)

            with self._lock:
                if token is not None:
                    token_int = int(token)
                    self._symbol_by_token[token_int] = canonical
                    self._token_to_symbol[token_int] = canonical
                    self._symbol_to_token[canonical] = token_int
                    self._token_by_symbol[canonical] = token_int

                self._latest_ticks[canonical] = payload
                self._tick_cache[canonical] = payload
                self._last_tick_time[canonical] = now
                self._last_tick_wallclock[canonical] = now
                self._last_tick_source[canonical] = "ws"
                self._symbols_with_tick.add(canonical)
                self._ticks_received_per_symbol[canonical] += 1

                if self._is_nifty_spot_tick(canonical, token):
                    self._spot_ws_first_tick_seen = True
                    self._latest_ticks["NSE:NIFTY"] = payload
                    self._tick_cache["NSE:NIFTY"] = payload
                    self._last_tick_time["NSE:NIFTY"] = now
                    self._last_tick_wallclock["NSE:NIFTY"] = now
                    self._last_tick_source["NSE:NIFTY"] = "ws"
                    self._symbols_with_tick.add("NSE:NIFTY")
                    if not getattr(self, "_spot_ws_first_tick_seen_logged", False):
                        self._spot_ws_first_tick_seen_logged = True
                        self._logger.info(
                            "MDM_WS_FIRST_TICK symbol=NSE:NIFTY token=%s ltp=%s",
                            token,
                            price,
                            extra={
                                "event": "MDM_WS_FIRST_TICK",
                                "symbol": "NSE:NIFTY",
                                "token": token,
                                "ltp": price,
                                "source": "ws_fast_cache",
                            },
                        )
        except Exception as e:
            self._logger.error("Failure in MarketDataManager._record_ws_arrival_fast: %s", e)

    def process_ticks(self, ticks: list[dict[str, Any]]) -> None:
        """Batch-enqueue WS ticks from KiteTicker callback.

        Called by WebSocketManager._on_ticks() for every tick batch —
        runs on the KiteTicker OS thread.  Under NO circumstances may
        this method block or raise; its only job is to hand each tick
        to the async queue.  All processing happens in _consume_ticks
        (asyncio task) or _tick_worker_loop (thread fallback).

        Args:
            ticks: Raw tick dicts from KiteTicker (instrument_token + last_price).
        Returns: None. Raises: None.
        """
        if not ticks:
            return

        try:
            iterator = iter(ticks)
        except TypeError:
            return
        for tick in iterator:
            if not isinstance(tick, dict):
                continue
            try:
                token_raw = tick.get("instrument_token")
                token_int = int(token_raw) if token_raw is not None else None
                symbol_resolved = None
                if token_int is not None:
                    with self._lock:
                        symbol_resolved = self._symbol_by_token.get(token_int)
                if symbol_resolved:
                    self._record_ws_arrival_fast(
                        symbol=symbol_resolved,
                        token=token_int,
                        ltp=tick.get("last_price", tick.get("ltp")),
                        raw_tick=tick,
                    )
                    now = time.monotonic()
                    last_logged = float(self._ws_raw_tick_log_at.get(symbol_resolved, 0.0))
                    if last_logged <= 0.0 or (now - last_logged) >= 60.0:
                        self._ws_raw_tick_log_at[symbol_resolved] = now
                        self._logger.info(
                            "WS_RAW_TICK_RECEIVED token=%s symbol_resolved=%s ltp=%s",
                            token_int,
                            symbol_resolved,
                            tick.get("last_price", tick.get("ltp")),
                            extra={
                                "event": "WS_RAW_TICK_RECEIVED",
                                "token": token_int,
                                "symbol_resolved": symbol_resolved,
                                "ltp": tick.get("last_price", tick.get("ltp")),
                            },
                        )
            except Exception:
                pass
            try:
                self._enqueue_tick_threadsafe(tick)
            except Exception as exc:  # noqa: BLE001 — WS thread must not raise
                self._logger.debug("process_ticks enqueue failed: %s", exc)

    def update_authoritative_ticks(self, ticks: list[dict[str, Any]]) -> None:
        """Update authoritative wall-clock tick snapshot. Args: ticks. Returns: None. Raises: None."""

        now = time.time()
        with self._authoritative_tick_lock:
            self.last_tick_time = now
            self.latest_ticks = list(ticks)

    def is_data_stale(self) -> bool:
        """Return whether feed age exceeds threshold. Args: none. Returns: bool. Raises: None."""

        with self._authoritative_tick_lock:
            return (time.time() - self.last_tick_time) > self._stale_threshold_seconds

    def data_age_ms(self) -> int:
        """Return authoritative feed age in milliseconds. Args: none. Returns: age ms. Raises: none."""

        now = time.time()
        with self._authoritative_tick_lock:
            last_authoritative = float(self.last_tick_time or 0.0)
        if last_authoritative > 0.0:
            return int(max(0.0, (now - last_authoritative) * 1000.0))
        with self._lock:
            tick_wallclock_values = list(self._last_tick_wallclock.values())
            tick_arrival_values = list(self._last_tick_time.values())
        candidates = [float(ts) for ts in tick_wallclock_values if float(ts) > 0.0]
        candidates.extend(float(ts) for ts in tick_arrival_values if float(ts) > 0.0)
        if not candidates:
            return 1_000_000_000
        freshest = max(candidates)
        return int(max(0.0, (now - freshest) * 1000.0))


    def _resolve_symbol_key_safe(self, symbol: str | int) -> str | None:
        """Resolve symbol/token aliases to canonical symbol. Args: symbol. Returns: canonical or None. Raises: none."""

        try:
            if isinstance(symbol, int):
                with self._lock:
                    resolved = self._symbol_by_token.get(int(symbol))
                return self._canonical_symbol(resolved) if resolved else None
            raw = str(symbol).strip()
            if not raw:
                return None
            if raw.isdigit():
                with self._lock:
                    resolved = self._symbol_by_token.get(int(raw))
                return self._canonical_symbol(resolved) if resolved else None
            return self._canonical_symbol(raw)
        except Exception:
            return None

    def symbol_data_age_ms(self, symbol: str | int) -> int:
        """Return age in milliseconds for one symbol/token. Args: symbol/token. Returns: age ms. Raises: none."""

        resolved_symbol = self._resolve_symbol_key_safe(symbol)
        if not resolved_symbol:
            return 1_000_000_000
        now = time.time()
        with self._lock:
            wallclock = float(self._last_tick_wallclock.get(resolved_symbol, 0.0) or 0.0)
            arrival = float(self._last_tick_time.get(resolved_symbol, 0.0) or 0.0)
        freshest = max(wallclock, arrival)
        if freshest <= 0.0:
            return 1_000_000_000
        return int(max(0.0, (now - freshest) * 1000.0))

    def symbol_data_age_ms_or_none(self, symbol: str | int) -> int | None:
        """Return age in milliseconds or None when no tick exists. Args: symbol/token. Returns: age ms or none. Raises: none."""

        resolved_symbol = self._resolve_symbol_key_safe(symbol)
        now = time.time()
        candidates: list[str] = []
        if str(symbol).strip().upper() in {"NIFTY", "NSE:NIFTY", "NIFTY50", "NSE:NIFTY50"}:
            candidates.append("NSE:NIFTY")
            with self._lock:
                token = self._token_by_symbol.get("NSE:NIFTY") or NIFTY_SPOT_TOKEN
                if token:
                    mapped = self._symbol_by_token.get(int(token))
                    if mapped:
                        candidates.append(mapped)
        elif resolved_symbol:
            candidates.append(resolved_symbol)
        if not candidates:
            return None
        with self._lock:
            freshest = 0.0
            for candidate in set(candidates):
                wallclock = float(self._last_tick_wallclock.get(candidate, 0.0) or 0.0)
                arrival = float(self._last_tick_time.get(candidate, 0.0) or 0.0)
                freshest = max(freshest, wallclock, arrival)
        if freshest <= 0.0:
            return None
        return int(max(0.0, (now - freshest) * 1000.0))

    def symbol_has_tick(self, symbol: str | int) -> bool:
        """Return whether at least one tick exists for symbol/token. Args: symbol/token. Returns: bool. Raises: none."""

        resolved_symbol = self._resolve_symbol_key_safe(symbol)

        if str(symbol).strip().upper() in {"NIFTY", "NSE:NIFTY", "NIFTY50", "NSE:NIFTY50"}:
            candidates = ["NSE:NIFTY"]
            with self._lock:
                token = self._token_by_symbol.get("NSE:NIFTY") or NIFTY_SPOT_TOKEN
                if token:
                    mapped = self._symbol_by_token.get(int(token))
                    if mapped:
                        candidates.append(mapped)
                for candidate in set(candidates):
                    if candidate in self._latest_ticks:
                        return True
                    if float(self._last_tick_wallclock.get(candidate, 0.0) or 0.0) > 0:
                        return True
                    if float(self._last_tick_time.get(candidate, 0.0) or 0.0) > 0:
                        return True

        if not resolved_symbol:
            return False

        with self._lock:
            if resolved_symbol in self._latest_ticks:
                return True
            if float(self._last_tick_wallclock.get(resolved_symbol, 0.0) or 0.0) > 0.0:
                return True
            if float(self._last_tick_time.get(resolved_symbol, 0.0) or 0.0) > 0.0:
                return True
        return False


    # ------------------------------------------------------------------
    # Internal plumbing

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        """Deprecated sync entrypoint retained for compatibility."""
        self._enqueue_tick_threadsafe(tick)

    async def _consume_ticks(self) -> None:
        """Consume websocket ticks serially and build finalized candles."""
        while True:
            raw = await self._tick_queue.get()
            try:
                self._process_queued_tick(raw)
            except Exception as exc:
                preview = {}
                if isinstance(raw, dict):
                    preview = {
                        "keys": sorted(list(raw.keys()))[:30],
                        "instrument_token": raw.get("instrument_token"),
                        "token": raw.get("token"),
                        "symbol": raw.get("symbol"),
                        "last_price": raw.get("last_price"),
                        "ltp": raw.get("ltp"),
                        "timestamp": str(raw.get("timestamp")),
                        "exchange_timestamp": str(raw.get("exchange_timestamp")),
                    }

                log_throttled(
                    self._logger,
                    "mdm_tick_consumer_error",
                    "MDM_TICK_CONSUMER_ERROR error=%s raw_preview=%s"
                    % (repr(exc), preview),
                    interval_sec=5.0,
                    level=logging.ERROR,
                    exc_info=True,
                )
            finally:
                try:
                    self._tick_queue.task_done()
                except Exception:
                    pass

    def _drain_tick_queue_sync(self) -> None:
        """Drain queued ticks serially when async loop is unavailable."""
        while True:
            try:
                raw = self._tick_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._process_queued_tick(raw)

    def _normalize_ws_tick(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize websocket tick payload. Args: raw. Returns: canonical tick or None. Raises: none."""
        if not isinstance(raw, dict):
            return None

        token_raw = raw.get("instrument_token") or raw.get("token")
        if token_raw is None:
            return None

        try:
            token = int(token_raw)
        except Exception:
            return None

        with self._lock:
            symbol = (
                raw.get("symbol")
                or self._symbol_by_token.get(token)
                or self._token_to_symbol.get(token)
            )

        if not symbol:
            log_throttled(
                self._logger,
                f"mdm_unmapped_token_{token}",
                "MDM_TICK_DROPPED reason=unmapped_token token=%s" % token,
                interval_sec=30.0,
                level=logging.WARNING,
            )
            return None

        price_raw = raw.get("last_price") or raw.get("ltp") or raw.get("price")
        if price_raw is None:
            return None

        try:
            price = float(price_raw)
        except Exception:
            return None

        if price <= 0:
            return None

        ts_raw = raw.get("exchange_timestamp") or raw.get("timestamp") or raw.get("received_at")
        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(ts) or ts.year < 2020:
            ts = pd.Timestamp.utcnow()

        return {
            **to_json_safe(dict(raw)),
            "symbol": str(symbol),
            "instrument_token": token,
            "token": token,
            "ltp": price,
            "last_price": price,
            "timestamp": ts.isoformat(),
            "timestamp_ms": float(pd.Timestamp(ts).timestamp() * 1000.0),
            "source": raw.get("source") or "ws",
            "received_at": float(raw.get("received_at") or time.time()),
        }

    def _process_queued_tick(self, raw: dict[str, Any]) -> None:
        raw = self._normalize_ws_tick(raw)
        if raw is None:
            return
        enqueued_mono = raw.get("_enqueued_monotonic")
        if isinstance(enqueued_mono, (int, float)):
            self._event_loop_lag_seconds = max(0.0, time.monotonic() - float(enqueued_mono))
        if not raw.get("symbol"):
            token = raw.get("instrument_token")
            if token is None:
                return

            token_int = int(token)
            with self._lock:
                symbol_from_map = self._symbol_by_token.get(token_int)

            if not symbol_from_map:
                log_throttled(
                    self._logger,
                    f"unmapped_token_{token_int}",
                    "Unmapped token %s — tick dropped" % token_int,
                    interval_sec=30.0,
                    level=logging.WARNING,
                )
                return
            raw = {**raw, "symbol": symbol_from_map}

        try:
            tick = validate_tick(raw)
        except DataIntegrityError as exc:
            self._logger.debug("Tick validation failed: %s — dropped", exc)
            return

        symbol = tick.symbol
        tick_ts = pd.to_datetime(tick.timestamp, utc=True, errors="coerce")
        if pd.isna(tick_ts):
            return

        last_ts = self._last_tick_ts.get(symbol)
        if last_ts is not None:
            last_ts = pd.to_datetime(last_ts, utc=True, errors="coerce")
            if not pd.isna(last_ts) and tick_ts < last_ts:
                self._logger.debug(
                    "MDM_TICK_DROPPED reason=older_timestamp symbol=%s ts=%s last=%s",
                    symbol,
                    tick_ts,
                    last_ts,
                )
                return

            if not pd.isna(last_ts) and tick_ts == last_ts:
                last_snapshot = self._last_tick_snapshot.get(symbol, {})
                same_price = float(last_snapshot.get("ltp", -1)) == float(tick.ltp)
                same_volume = last_snapshot.get("volume") == raw.get("volume_traded_today")

                if same_price and same_volume:
                    self._logger.debug(
                        "MDM_TICK_DROPPED reason=duplicate_tick symbol=%s ts=%s",
                        symbol,
                        tick_ts,
                    )
                    return

        self._last_tick_snapshot[symbol] = {
            "ltp": tick.ltp,
            "volume": raw.get("volume_traded_today") or raw.get("volume_traded"),
        }
        self._last_tick_ts[symbol] = tick_ts
        engine = self._get_engine(symbol)
        candle = engine.on_tick(tick)
        # ── EMIT TICK: replaces bare _store_tick call ─────────────────────────
        # _emit_tick calls _store_tick internally AND dispatches to _subscribers.
        # Before this fix: DataHub.subscribe_ticks() callbacks (including
        # StrategyRunner's per-symbol callbacks) were registered in _subscribers
        # but NEVER called from the WS path — _store_tick only cached the tick.
        # _emit_tick is the correct call; it was defined but never invoked.
        tick_dict = tick.to_dict()
        tick_dict["timestamp"] = pd.to_datetime(
            tick_dict["timestamp"], utc=True, errors="coerce"
        ).isoformat()
        token_value = raw.get("instrument_token") or raw.get("token")
        if token_value is not None:
            tick_dict["instrument_token"] = token_value
        price_value = raw.get("last_price") or raw.get("ltp") or tick.ltp
        tick_dict["last_price"] = price_value
        tick_dict["ltp"] = tick.ltp
        tick_dict["source"] = "ws"
        tick_dict["received_at"] = raw.get("received_at", time.time())
        if raw.get("exchange_timestamp") is not None:
            tick_dict["exchange_timestamp"] = raw.get("exchange_timestamp")
        if raw.get("volume_traded") is not None:
            tick_dict["volume_traded"] = raw.get("volume_traded")
        if raw.get("volume_traded_today") is not None:
            tick_dict["volume_traded_today"] = raw.get("volume_traded_today")
        if raw.get("oi") is not None:
            tick_dict["oi"] = raw.get("oi")

        # Structured Logging for tick received (Objective 6)
        try:
            self._logger.debug(
                "TICK_RECEIVED",
                extra={
                    "event": "tick_received",
                    "symbol": symbol,
                    "price": tick.ltp,
                    "source": "ws",
                    "token": token_value,
                },
            )
        except Exception:
            pass

        normalized_live = self.normalize_live_tick(tick_dict, source="ws")
        if normalized_live is None:
            return
        self._ingest_normalized_tick(normalized_live)
        if candle:
            bar = {
                "symbol": symbol,
                "timestamp": candle["timestamp"] if isinstance(candle, dict) else candle.timestamp,
                "open": float(candle["open"] if isinstance(candle, dict) else candle.open),
                "high": float(candle["high"] if isinstance(candle, dict) else candle.high),
                "low": float(candle["low"] if isinstance(candle, dict) else candle.low),
                "close": float(candle["close"] if isinstance(candle, dict) else candle.close),
                "volume": int(float((candle.get("volume", 0) if isinstance(candle, dict) else getattr(candle, "volume", 0)) or 0)),
                "source": "ws_candle",
            }
            self._ohlc[symbol].append(bar)
            self._publish_closed_bar(bar)
            self._logger.info(
                "LIVE_CANDLE_CLOSED symbol=%s source=ws_candle close=%s",
                symbol,
                bar["close"],
            )

        # ── PIPELINE FEED ─────────────────────────────────────────────────────
        # Feed the deterministic MarketDataPipeline so pipeline.candles_ready()
        # and pipeline.get_candles() reflect live data.  Lazy import avoids
        # circular dependency.  Errors are caught so MDM consumer loop never dies.
        try:
            from nifty_scalper_bot.data.pipeline import get_pipeline  # noqa: PLC0415
            _pipeline_candle = get_pipeline().on_tick(raw)
            if _pipeline_candle is not None:
                self._logger.debug(
                    "CANDLE_FORMED symbol=%s ts=%s open=%.2f high=%.2f low=%.2f close=%.2f",
                    _pipeline_candle.symbol,
                    _pipeline_candle.timestamp,
                    _pipeline_candle.open,
                    _pipeline_candle.high,
                    _pipeline_candle.low,
                    _pipeline_candle.close,
                )
        except Exception as _pipe_exc:  # pragma: no cover
            self._logger.debug("pipeline.on_tick feed failed: %s", _pipe_exc)

    def _get_engine(self, symbol: str) -> CandleEngine:
        """Return or create candle engine for symbol."""
        if symbol not in self._engines:
            self._engines[symbol] = CandleEngine()
        return self._engines[symbol]

    def _seed_mapping(self, symbol: str, token: int | None) -> None:
        if token is None:
            return
        try:
            token_int = int(token)
        except (TypeError, ValueError):
            return
        self.register_symbol(symbol, token_int)

    def register_symbol(self, symbol: str, token: int) -> None:
        """Args: symbol/token; Returns: none; Raises: RuntimeError on bad data."""

        normalized_symbol = self._canonical_symbol(symbol)
        token_int = int(token)
        with self._lock:
            if self._token_by_symbol.get(normalized_symbol) == token_int:
                return
            self._token_by_symbol[normalized_symbol] = token_int
            self._symbol_by_token[token_int] = normalized_symbol
            self._symbol_to_token[normalized_symbol] = token_int
            self._token_to_symbol[token_int] = normalized_symbol
        try:
            if self._resolver is not None and hasattr(self._resolver, "register"):
                self._resolver.register(normalized_symbol, token_int)
            datahub = getattr(self, "_datahub", None)
            if datahub is not None:
                if hasattr(datahub, "register_local_symbol"):
                    datahub.register_local_symbol(normalized_symbol, token_int)
                elif hasattr(datahub, "register_symbol"):
                    datahub.register_symbol(normalized_symbol, token_int)
            if normalized_symbol == "NSE:NIFTY":
                self._logger.info(
                    "SPOT_TOKEN_RESOLVED symbol=%s token=%s",
                    normalized_symbol,
                    token_int,
                    extra={
                        "event": "SPOT_TOKEN_RESOLVED",
                        "symbol": normalized_symbol,
                        "token": token_int,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Resolver/DataHub sync skipped: %s", exc, exc_info=exc)

    def ws_status_snapshot(self) -> dict[str, Any]:
        """Return websocket runtime status. Args: none. Returns: status map. Raises: none."""

        ws = self._ws
        if ws is not None and hasattr(ws, "status_snapshot"):
            try:
                return dict(ws.status_snapshot())
            except Exception:
                self._logger.debug("ws.status_snapshot failed", exc_info=True)
        return {
            "running": False,
            "connected": bool(getattr(self, "_ws_connected", False)),
            "state": "disconnected",
            "tokens": 0,
            "connect_task_alive": False,
            "watchdog_task_alive": False,
            "last_tick_age_s": None,
            "last_pong_age_s": None,
            "circuit_open": False,
        }

    def _store_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        """Persist normalized *tick* for *symbol* and refresh derived series."""

        tick.setdefault("received_at", time.time())
        wallclock = self._tick_wallclock(tick) or time.time()

        # 🔥 PRODUCTION FIX — enforce canonical key
        symbol = self._canonical_symbol(symbol)

        cached_tick = dict(tick)
        token_value = cached_tick.get("instrument_token") or cached_tick.get("token")
        token_int = int(token_value) if token_value is not None else None
        now_wall = time.time()
        exchange_ts = self._tick_wallclock(cached_tick) or now_wall
        with self._lock:
            if token_int is not None:
                self._symbol_by_token[token_int] = symbol
                self._token_to_symbol[token_int] = symbol
                self._symbol_to_token[symbol] = token_int
                self._token_by_symbol[symbol] = token_int
            self._latest_ticks[symbol] = cached_tick
            self._tick_cache[symbol] = cached_tick
            self._last_tick_time[symbol] = float(exchange_ts)
            if "ltp" not in cached_tick or cached_tick.get("timestamp") is None:
                self._logger.debug(
                    "Condition met: mdm_history_append_rejected",
                    extra={"event": "mdm_history_append_rejected", "symbol": symbol},
                )
                return
            self._history[symbol].append(cached_tick)
            self._ticks_received_per_symbol[symbol] += 1
            self._symbols_with_tick.add(symbol)
            self._last_tick_wallclock[symbol] = float(now_wall)
            self._last_quote_ts_ms[symbol] = self._now_ms()
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

    def _publish_tick_to_bus_safe(self, tick: dict[str, Any]) -> None:
        """Publish a normalized tick to MessageBus without ever crashing MDM."""
        bus = getattr(self, "bus", None) or getattr(self, "_tick_bus", None)
        if bus is None:
            log_throttled(
                self._logger,
                "mdm_bus_publish_skipped_no_bus",
                "MDM_BUS_PUBLISH_SKIPPED reason=no_bus symbol=%s" % tick.get("symbol"),
                interval_sec=30.0,
                level=logging.DEBUG,
            )
            return
        if not self._bus_is_running():
            log_throttled(
                self._logger,
                "mdm_bus_publish_skipped_bus_not_running",
                "MDM_BUS_PUBLISH_SKIPPED reason=bus_not_running symbol=%s"
                % tick.get("symbol"),
                interval_sec=30.0,
                level=logging.DEBUG,
            )
            return
        loop = (
            getattr(self, "_main_loop", None)
            or getattr(self, "_event_loop", None)
            or getattr(self, "event_loop", None)
        )
        if loop is None or loop.is_closed():
            log_throttled(
                self._logger,
                "mdm_bus_publish_skipped_loop_unavailable",
                "MDM_BUS_PUBLISH_SKIPPED reason=loop_unavailable symbol=%s"
                % tick.get("symbol"),
                interval_sec=30.0,
                level=logging.WARNING,
            )
            return
        payload = to_json_safe(dict(tick))
        ts = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts) or getattr(ts, "year", 1970) < 2020:
            ts = pd.Timestamp.utcnow()
        payload["timestamp"] = pd.Timestamp(ts).isoformat()
        payload["timestamp_ms"] = float(pd.Timestamp(ts).timestamp() * 1000.0)
        payload["received_at"] = float(payload.get("received_at") or time.time())
        try:
            msg = Message(
                type=MessageType.TICK,
                data=payload,
                source="market_data_manager",
                timestamp=time.time(),
            )
            future = asyncio.run_coroutine_threadsafe(bus.publish(msg), loop)

            def _done(fut: Any) -> None:
                try:
                    fut.result()
                except Exception as exc:
                    log_throttled(
                        self._logger,
                        "mdm_bus_publish_failed",
                        "MDM_BUS_PUBLISH_FAILED symbol=%s error=%r"
                        % (tick.get("symbol"), exc),
                        interval_sec=10.0,
                        level=logging.ERROR,
                    )

            future.add_done_callback(_done)
        except Exception as exc:
            log_throttled(
                self._logger,
                "mdm_bus_publish_submit_failed",
                "MDM_BUS_PUBLISH_SUBMIT_FAILED symbol=%s error=%r"
                % (tick.get("symbol"), exc),
                interval_sec=10.0,
                level=logging.ERROR,
            )

    def _bus_is_running(self) -> bool:
        """Check message bus running status safely. Args: none. Returns: bool. Raises: none."""
        bus = getattr(self, "bus", None) or getattr(self, "_tick_bus", None)
        if bus is None:
            return False
        running_attr = getattr(bus, "running", None)
        if isinstance(running_attr, bool):
            return running_attr
        if callable(running_attr):
            try:
                return bool(running_attr())
            except Exception:
                return False
        is_running_method = getattr(bus, "is_running", None)
        if callable(is_running_method):
            try:
                return bool(is_running_method())
            except Exception:
                return False
        return bool(getattr(bus, "_running", False))


    def _dispatch_awaitable_callback_result(
        self,
        result: Any,
        *,
        symbol: str,
        callback: Any,
    ) -> None:
        """Schedule awaitable callback results. Args: result/symbol/callback. Returns: none. Raises: none."""
        if result is None or not inspect.isawaitable(result):
            return

        async def _await_result(awaitable: Any) -> None:
            try:
                await awaitable
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_throttled(
                    self._logger,
                    f"tick_async_callback_failed_{symbol}",
                    "Async tick callback failed symbol=%s callback=%s error=%r"
                    % (symbol, getattr(callback, "__qualname__", repr(callback)), exc),
                    interval_sec=10.0,
                    level=logging.ERROR,
                    exc_info=True,
                )

        loop = getattr(self, "_main_loop", None) or getattr(self, "_event_loop", None)
        if loop is not None and loop.is_running():
            try:
                if loop is asyncio.get_running_loop():
                    loop.create_task(_await_result(result))
                else:
                    asyncio.run_coroutine_threadsafe(_await_result(result), loop)
                return
            except RuntimeError:
                pass

        try:
            running_loop = asyncio.get_running_loop()
            running_loop.create_task(_await_result(result))
            return
        except RuntimeError:
            log_throttled(
                self._logger,
                f"tick_async_callback_dropped_{symbol}",
                "Async tick callback dropped symbol=%s reason=no_running_loop callback=%s"
                % (symbol, getattr(callback, "__qualname__", repr(callback))),
                interval_sec=10.0,
                level=logging.WARNING,
            )

    def _ingest_normalized_tick(self, tick: Mapping[str, Any]) -> None:
        """Ingest one canonical live tick. Args: tick; Returns: none; Raises: none."""
        symbol = str(tick["symbol"])
        source = str(tick.get("source") or "unknown").lower()
        tick_payload = dict(tick)
        self._emit_tick(symbol, tick_payload, source=source)

    def _emit_tick(self, symbol: str, tick: dict[str, Any], *, source: str) -> None:
        source = str(source or "unknown").lower()
        if source == "ws":
            self._last_ws_tick_mono = time.monotonic()
        self._store_tick(symbol, tick)
        if source == "ws":
            try:
                token_value = tick.get("instrument_token") or tick.get("token")
                token_int = int(token_value) if token_value is not None else None
                now = time.monotonic()
                last_logged = float(self._ws_stored_tick_log_at.get(symbol, 0.0))
                if last_logged <= 0.0 or (now - last_logged) >= 60.0:
                    self._ws_stored_tick_log_at[symbol] = now
                    self._logger.debug(
                        "MDM_TICK_CACHE_UPDATED symbol=%s token=%s source=ws",
                        symbol,
                        token_int,
                        extra={"event": "MDM_TICK_CACHE_UPDATED", "symbol": symbol, "token": token_int, "source": "ws"},
                    )
                    self._logger.debug(
                        "WS_TICK_STORED symbol=%s token=%s ltp=%s source=ws age_ms=0",
                        symbol,
                        token_int,
                        tick.get("ltp") or tick.get("last_price") or tick.get("price"),
                        extra={
                            "event": "WS_TICK_STORED",
                            "symbol": symbol,
                            "token": token_int,
                            "ltp": tick.get("ltp")
                            or tick.get("last_price")
                            or tick.get("price"),
                            "source": "ws",
                            "age_ms": 0,
                        },
                    )
            except Exception:
                pass
        callbacks: list[TickCallback]
        tick_payload = to_json_safe(dict(tick))
        ts_payload = pd.to_datetime(tick_payload.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts_payload) or ts_payload.year < 2020:
            ts_payload = pd.Timestamp.utcnow()
        tick_payload["timestamp"] = ts_payload.isoformat()
        tick_payload["timestamp_ms"] = float(pd.Timestamp(ts_payload).timestamp() * 1000.0)
        tick_payload["received_at"] = float(tick_payload.get("received_at") or time.time())
        with self._lock:
            self._last_tick_source[symbol] = source
            callbacks = list(self._subscribers.get(symbol, ()))
            self._tick_counter += 1
        self._publish_tick_to_bus_safe(
            {
                **dict(tick_payload),
                "symbol": symbol,
                "source": source,
                "trace_id": tick_payload.get("trace_id")
                or f"{symbol}-{time.monotonic_ns()}",
            }
        )
        subscriber_count = len(callbacks)
        if subscriber_count == 0 and source == "ws":
            selected_or_active = (
                symbol in self._active_subscribed_symbols
                or symbol in self._tracked_symbols
            )
            first_seen_map = getattr(self, "_first_seen_at_by_symbol", None)
            if first_seen_map is None:
                first_seen_map = {}
                self._first_seen_at_by_symbol = first_seen_map
            first_seen_at = first_seen_map.get(symbol)
            if first_seen_at is None:
                first_seen_map[symbol] = time.monotonic()
            else:
                age = time.monotonic() - first_seen_at
                if selected_or_active and age > 30:
                    bus_running = self._bus_is_running()
                    if bus_running:
                        log_throttled(
                            self._logger,
                            f"mdm_tick_bus_only_delivery_{symbol}",
                            "MDM_TICK_BUS_ONLY_DELIVERY symbol=%s age_s=%.1f"
                            % (symbol, age),
                            interval_sec=120,
                            level=logging.DEBUG,
                            extra={
                                "event": "MDM_TICK_BUS_ONLY_DELIVERY",
                                "symbol": symbol,
                                "age_s": age,
                            },
                        )
                    else:
                        log_throttled(
                            self._logger,
                            f"mdm_tick_no_direct_subscriber_{symbol}",
                            "MDM_TICK_NO_DIRECT_SUBSCRIBER symbol=%s age_s=%.1f note=bus_unavailable"
                            % (symbol, age),
                            interval_sec=60,
                            level=logging.WARNING,
                            extra={
                                "event": "MDM_TICK_NO_DIRECT_SUBSCRIBER",
                                "symbol": symbol,
                                "age_s": age,
                            },
                        )
        _price_val = tick.get("ltp") or tick.get("last_price") or tick.get("price")
        _token_val = tick.get("instrument_token") or tick.get("token")
        try:
            self._logger.debug(
                "MDM_TICK_RECEIVED symbol=%s source=%s subscribers=%d",
                symbol,
                source,
                subscriber_count,
                extra={
                    "event": "MDM_TICK_RECEIVED",
                    "symbol": symbol,
                    "token": _token_val,
                    "price": _price_val,
                    "source": source,
                    "has_subscribers": subscriber_count > 0,
                    "subscriber_count": subscriber_count,
                    "emitted": subscriber_count > 0,
                    "reason": "tick_received",
                },
            )
            if source == "ws" and _token_val is not None:
                token_int = int(_token_val)
                first_for_token = token_int not in self._confirmed_subscriptions
                self._confirmed_subscriptions.add(token_int)
                if first_for_token:
                    self._logger.info(
                        "MDM_WS_FIRST_TICK symbol=%s token=%s ltp=%s",
                        symbol,
                        token_int,
                        _price_val,
                        extra={
                            "event": "MDM_WS_FIRST_TICK",
                            "symbol": symbol,
                            "token": token_int,
                            "ltp": _price_val,
                        },
                    )
                self._pending_subscriptions.discard(token_int)
                self._pending_subscription_tokens.discard(token_int)
            if source == "ws" and symbol == "NSE:NIFTY":
                if not self._spot_ws_first_tick_seen:
                    self._spot_ws_first_tick_seen = True
                tick_age = 0.0
                try:
                    tick_age = max(time.time() - float(self._last_tick_time.get(symbol, 0.0) or 0.0), 0.0)
                except Exception:
                    tick_age = 0.0
                log_throttled(
                    self._logger,
                    "mdm_ws_tick_received_nse_nifty",
                    "MDM_WS_TICK_RECEIVED symbol=NSE:NIFTY",
                    interval_sec=60.0,
                    level=logging.DEBUG,
                    extra={"event": "MDM_WS_TICK_RECEIVED", "symbol": "NSE:NIFTY", "age": round(tick_age, 3)},
                )
        except Exception:  # pragma: no cover - defensive
            pass
        self.bump_heartbeat()
        now_mono = time.monotonic()
        tick_stats_interval = float(os.getenv("TICK_STATS_INTERVAL", "5.0"))
        if now_mono - self._last_tick_log_time >= tick_stats_interval:
            with self._lock:
                if now_mono - self._last_tick_log_time >= tick_stats_interval:
                    self._last_tick_log_time = now_mono
                    cached_count = len(self._tick_cache)
                    tick_count = self._tick_counter
                    self._tick_counter = 0
                    rate_interval = max(
                        now_mono - self._last_tick_rate_snapshot_at,
                        1e-6,
                    )
                    rates = {
                        sym: (
                            count - self._last_tick_rate_snapshot.get(sym, 0)
                        )
                        / rate_interval
                        for sym, count in self._ticks_received_per_symbol.items()
                        if sym in self._active_subscribed_symbols
                    }
                    history_lengths = {
                        sym: len(self._history.get(sym, ()))
                        for sym in sorted(self._active_subscribed_symbols)
                    }
                    self._last_tick_rate_snapshot = dict(
                        self._ticks_received_per_symbol
                    )
                    self._last_tick_rate_snapshot_at = now_mono
            self._logger.debug(
                "EVENT|tick_stats|cached=%d|ticks_last_5s=%d",
                cached_count,
                tick_count,
            )
            self._logger.debug(
                "Condition met: mdm_tick_health",
                extra={
                    "event": "mdm_tick_health",
                    "ticks_per_second_by_symbol": rates,
                    "history_length_by_symbol": history_lengths,
                },
            )
        if now_mono - self._last_tick_stats_log >= 120.0:
            self._last_tick_stats_log = now_mono
            with self._lock:
                live_symbols = len(self._symbols_with_tick)
                subscribed = len(self._active_subscribed_symbols)
                stale_symbols = sum(
                    1
                    for sym in self._active_subscribed_symbols
                    if (time.time() - float(self._last_tick_time.get(sym, 0.0) or 0.0))
                    >= self._option_stale_seconds
                )
            self._logger.info(
                "MARKET_DATA_HEALTH_SUMMARY ws_connected=%s subscribed=%d live_symbols=%d stale_symbols=%d poll_fallbacks=%d bus_running=%s",
                self._is_ws_connected(),
                subscribed,
                live_symbols,
                stale_symbols,
                int(getattr(self, "_poll_fallback_count", 0)),
                bool(getattr(getattr(self, "bus", None), "running", False)),
            )
        try:
            self._m_ticks.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    loop = self._main_loop
                    if loop is not None and loop.is_running():
                        # Default-argument capture prevents closure bug where
                        # all lambdas in the for-loop share the last `callback`.
                        loop.call_soon_threadsafe(
                            lambda _cb=callback, _tp=tick_payload: safe_task(_cb(dict(_tp)))
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
                    result = callback(dict(tick_payload))
                    self._dispatch_awaitable_callback_result(
                        result,
                        symbol=symbol,
                        callback=callback,
                    )
            except Exception as exc:
                log_throttled(
                    self._logger,
                    f"tick_callback_failed_{symbol}",
                    "Tick callback failed symbol=%s error=%r" % (symbol, exc),
                    interval_sec=10.0,
                    level=logging.ERROR,
                    exc_info=True,
                )
        try:
            delivered = subscriber_count > 0
            direct_subscribers = subscriber_count
            bus_publish_attempted = bool(getattr(self, "bus", None))
            bus_running = bool(getattr(getattr(self, "bus", None), "running", False))
            self._logger.debug(
                "MDM_TICK_EMITTED symbol=%s delivered=%s subscribers=%d source=%s",
                symbol,
                delivered,
                subscriber_count,
                source,
                extra={
                    "event": "MDM_TICK_EMITTED",
                    "symbol": symbol,
                    "token": _token_val,
                    "delivered_to_subscribers": delivered,
                    "subscriber_count": subscriber_count,
                    "source": source,
                    "emitted": delivered,
                    "reason": "subscribers_present" if delivered else "no_subscribers",
                },
            )
            # Emit a single INFO-level event the FIRST time a subscriber
            # receives (or fails to receive) a tick for a symbol — this is
            # the critical breakpoint that reveals whether the MDM→DataHub
            # bridge is live for that symbol, without flooding per-tick.
            first_emit_log_tracker = getattr(self, "_first_emit_logged", None)
            if first_emit_log_tracker is None:
                first_emit_log_tracker = set()
                self._first_emit_logged = first_emit_log_tracker
            emit_key = (symbol, delivered)
            if emit_key not in first_emit_log_tracker:
                first_emit_log_tracker.add(emit_key)
                self._logger.info(
                    "MDM_TICK_EMITTED_FIRST symbol=%s direct_delivered=%s direct_subscribers=%d bus_running=%s source=%s",
                    symbol,
                    delivered,
                    direct_subscribers,
                    bus_running,
                    source,
                    extra={
                        "event": "MDM_TICK_EMITTED",
                        "symbol": symbol,
                        "token": _token_val,
                        "direct_delivered_to_subscribers": delivered,
                        "direct_subscriber_count": direct_subscribers,
                        "bus_publish_attempted": bus_publish_attempted,
                        "bus_running": bus_running,
                        "source": source,
                        "first_emit": True,
                    },
                )
        except Exception:  # pragma: no cover - defensive
            pass

    async def warmup_history(
        self,
        symbols: list[str],
        lookback_minutes: int = 30,
    ) -> None:
        """Warm caches from minute candles; Args: symbols/lookback; Returns: None; Raises: RuntimeError."""

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=lookback_minutes)

            self._logger.info(
                "WARMUP_START symbols=%d lookback=%dmin",
                len(symbols),
                lookback_minutes,
            )

            rest_client = self._rest_client
            fetcher = getattr(rest_client, "get_historical_data", None)
            if not callable(fetcher):
                fetcher = getattr(rest_client, "historical_data", None)
            if not callable(fetcher):
                raise RuntimeError("WARMUP failed: historical data client unavailable")

            for symbol in symbols:
                canonical_symbol = normalize_symbol(str(symbol))
                token = self._token_by_symbol.get(canonical_symbol)
                if not token:
                    # BUG-β FIX: Never raise — skip and warn so remaining
                    # symbols are still warmed and callers don't abort.
                    self._logger.warning(
                        "WARMUP_SKIP: missing token for %s",
                        canonical_symbol,
                        extra={
                            "event": "warmup_skip_missing_token",
                            "symbol": canonical_symbol,
                        },
                    )
                    continue

                if asyncio.iscoroutinefunction(fetcher):
                    candles = await fetcher(
                        instrument_token=token,
                        from_date=start_time,
                        to_date=end_time,
                        interval="minute",
                    )
                else:
                    candles = await asyncio.to_thread(
                        fetcher,
                        token,
                        start_time,
                        end_time,
                        "minute",
                    )

                hydrated_count = 0
                for candle in candles or ():
                    candle_dt = candle.get("date")
                    if isinstance(candle_dt, str):
                        try:
                            candle_dt = datetime.fromisoformat(
                                candle_dt.replace("Z", "+00:00")
                            )
                        except ValueError:
                            continue
                    if not isinstance(candle_dt, datetime):
                        continue
                    if candle_dt.tzinfo is None:
                        candle_dt = candle_dt.replace(tzinfo=timezone.utc)

                    historical_tick = {
                        "symbol": canonical_symbol,
                        "ltp": float(candle.get("close", 0.0)),
                        "volume": float(candle.get("volume", 0.0)),
                        "timestamp": candle_dt,
                        "source": "historical",
                    }
                    validated = validate_tick(historical_tick)
                    engine = self._get_engine(canonical_symbol)
                    finalized = engine.on_tick(validated)
                    if finalized:
                        self._ohlc[self._bar_symbol_key(canonical_symbol)].append(
                            finalized
                        )
                    validated_ts = getattr(validated, "timestamp", None)
                    if validated_ts is not None:
                        self._last_historical_ts[canonical_symbol] = float(
                            validated_ts.timestamp()
                        )
                    hydrated_count += 1
                    self.ingest_historical_bar(
                        {
                            "symbol": canonical_symbol,
                            "open": float(candle.get("open", 0.0)),
                            "high": float(candle.get("high", 0.0)),
                            "low": float(candle.get("low", 0.0)),
                            "close": float(candle.get("close", 0.0)),
                            "volume": float(candle.get("volume", 0.0)),
                            "timestamp": candle_dt,
                        }
                    )
                self._logger.info("Hydrated %s: %d bars", canonical_symbol, hydrated_count)

            self._logger.info("WARMUP_COMPLETE")
        except Exception as e:
            self._logger.error("Failure in warmup_history: %s", e)
            raise

    def _is_ws_connected(self) -> bool:
        """Args: none; Returns: websocket connectivity; Raises: none."""

        try:
            if self._ws is None:
                return bool(self._ws_connected)
            return bool(self._ws.is_connected())
        except Exception:  # noqa: BLE001
            return bool(self._ws_connected)

    def _is_ws_healthy(self) -> bool:
        """Return WS health. Args: none. Returns: bool. Raises: None."""

        try:
            connected = self._is_ws_connected()
            now_mono = time.monotonic()
            last_tick_age = (
                max(0.0, now_mono - self._last_ws_tick_mono)
                if self._last_ws_tick_mono > 0
                else float("inf")
            )
            event_loop_lag = max(0.0, float(self._event_loop_lag_seconds))
            return connected and last_tick_age < 2.0 and event_loop_lag < 0.5
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _is_ws_healthy: %s", exc)
            return False

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
            except RuntimeError as exc:
                self._logger.critical(
                    "Failure in _health_monitor_loop: %s",
                    exc,
                    extra={"event": "mdm_health_monitor_circuit_break"},
                    exc_info=exc,
                )
                os._exit(1)
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
        self._monitor_spot_ws_health()
        if not self._is_ws_healthy():
            self._zombie_stale_logged = False
            return

        now = time.time()
        with self._lock:
            candidate_symbols = sorted(
                sym
                for sym in self._active_subscribed_symbols
                if sym in self._symbols_with_tick
            )
            stale_symbols = [
                sym
                for sym in candidate_symbols
                if (now - self._last_tick_time.get(sym, now))
                > self._zombie_tick_threshold_sec
            ]
        if not candidate_symbols:
            self._zombie_stale_logged = False
            return

        stale_ratio = len(stale_symbols) / max(len(candidate_symbols), 1)
        self._logger.info(
            "Condition met: mdm_zombie_summary",
            extra={
                "event": "mdm_zombie_summary",
                "active_symbols": len(candidate_symbols),
                "stale_symbols": len(stale_symbols),
                "stale_ratio": round(stale_ratio, 3),
                "threshold_seconds": self._zombie_tick_threshold_sec,
            },
        )
        if stale_ratio < 0.70:
            self._zombie_stale_logged = False
            # Recovery observed — reset exponential backoff counter
            # so the next genuine stall restarts at the base cooldown.
            if self._zombie_restart_consecutive:
                self._zombie_restart_consecutive = 0
            return

        if not self._zombie_stale_logged and stale_symbols:
            self._logger.critical(
                "CRITICAL zombie_tick_detected symbols=%s threshold=%.2fs",
                stale_symbols[:8],
                self._zombie_tick_threshold_sec,
                extra={
                    "event": "mdm_zombie_tick_detected",
                    "symbols": stale_symbols,
                    "stale_ratio": stale_ratio,
                },
            )
            self._zombie_stale_logged = True

        self._trigger_zombie_ws_restart()

    def _trigger_zombie_ws_restart(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        now = time.monotonic()
        if now < self._zombie_breaker_open_until:
            return

        # Exponential backoff between consecutive zombie restart
        # triggers.  WebSocketManager itself already does full-jitter
        # backoff on the reconnect, so this protects MDM against
        # flapping-trigger storms.
        cooldown = min(
            self._zombie_restart_cooldown_sec
            * (2 ** max(0, self._zombie_restart_consecutive - 1)),
            self._zombie_restart_cooldown_max_sec,
        )
        # Apply full jitter so multiple zombie detectors don't line up.
        cooldown *= 0.5 + uniform(0.0, 0.5)

        since_last = now - self._zombie_last_restart_attempt_at
        if since_last < cooldown:
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
                self._zombie_restart_consecutive += 1
                self._zombie_restart_attempts.append(now)
                while self._zombie_restart_attempts and (
                    now - self._zombie_restart_attempts[0]
                ) > 120.0:
                    self._zombie_restart_attempts.popleft()
                self._logger.warning(
                    "Condition met: mdm_zombie_ws_restart",
                    extra={
                        "event": "mdm_zombie_ws_restart",
                        "failure_count": self._zombie_restart_failures,
                        "consecutive": self._zombie_restart_consecutive,
                        "cooldown_sec": round(cooldown, 2),
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

        if len(self._zombie_restart_attempts) > 5:
            self._logger.critical(
                "Failure in _trigger_zombie_ws_restart: restart circuit breaker tripped",
                extra={
                    "event": "mdm_zombie_restart_circuit_breaker",
                    "attempts_in_2m": len(self._zombie_restart_attempts),
                },
            )
            raise RuntimeError("WebSocket restart circuit breaker exceeded: >5 in 120s")

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
        last_ws_healthy: bool | None = None
        last_poll_active: bool | None = None

        while not self._rest_poll_stop.is_set():
            loop_start = time.time()

            ws_healthy = self._is_ws_healthy()
            poll_active = bool(self._rest_poll_enabled) and (not ws_healthy)
            if ws_healthy and last_ws_healthy is not True:
                self._logger.info("WS HEALTHY")
            if poll_active and last_poll_active is not True:
                self._logger.info("POLLING ACTIVE")
            last_ws_healthy = ws_healthy
            last_poll_active = poll_active
            if not poll_active:
                time.sleep(1)
                continue

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
            now_dt = datetime.now(timezone.utc)
            if self._active_subscribed_symbols or self._tracked_symbols:
                ordered = self._poll_candidates(now_dt)
            else:
                ordered = [sym for sym in ordered if self._should_poll_symbol(sym, now_dt)]
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
    def ensure_tracking(
        self, symbol: str, *, seed: bool = True, subscribe: bool = True
    ) -> bool:
        """Ensure *symbol* is tracked for REST polling and optional seeding.

        Args:
            symbol: Trading symbol identifier to watch.
            seed: Flag indicating whether to seed the cache from broker REST.
            subscribe: Whether to request WS subscription intent for symbol.

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
            if subscribe and self._is_ws_healthy():
                self.request_symbol_subscription(sym)
            if seed and not self._seed_completed and not self._is_ws_healthy():
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
            if self._is_ws_healthy():
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

            now_ts = pd.Timestamp.utcnow()
            tick = {
                "symbol": symbol,
                "ltp": ltp,
                "last_price": ltp,
                "bid": bid,
                "ask": ask,
                "timestamp": now_ts.isoformat(),
                "timestamp_ms": float(now_ts.timestamp() * 1000.0),
                "received_at": time.time(),
                "source": "rest",
            }
            token = payload.get("instrument_token") or payload.get("token")
            if token is not None:
                tick["instrument_token"] = int(token)
                tick["token"] = int(token)

            if ltp is not None:
                self._emit_tick(symbol, tick, source="rest")
            else:
                with self._lock:
                    self._latest_ticks[symbol] = tick
                    self._last_tick_source[symbol] = "rest"
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
        canonical_symbol = self._canonical_symbol(symbol)
        try:
            latest = self.get_latest_tick(canonical_symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in get_quote: %s",
                exc,
                extra={
                    "event": "mdm_get_quote_failed",
                    "symbol": canonical_symbol,
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
            self._last_mid[canonical_symbol] = (mid, now_ms)
        self._last_quote_ts_ms[canonical_symbol] = now_ms
        return quote

    def get_symbol_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a unified MarketSnapshot for one symbol."""
        canonical = self._canonical_symbol(symbol)
        tick = self.get_latest_tick(canonical) or {}
        ltp = _coerce_float(tick.get("ltp") or tick.get("last_price"))
        bid = _coerce_float(tick.get("bid"))
        ask = _coerce_float(tick.get("ask"))
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (float(bid) + float(ask)) / 2.0
        age = None
        wallclock = self._tick_wallclock(tick)
        if wallclock is not None:
            age = max(0.0, time.time() - wallclock)
        with self._lock:
            source = str(
                self._last_tick_source.get(canonical) or tick.get("source") or "unknown"
            ).lower()
            recent_cutoff = time.time() - 60.0
            history = list(self._history.get(canonical, ()))
            real_ticks = sum(
                1
                for row in history
                if str(row.get("source") or source).lower() == "ws"
                and (self._tick_wallclock(row) or 0.0) >= recent_cutoff
            )
            bars = list(self._ohlc.get(self._bar_symbol_key(canonical), ()))
        if ltp is None:
            latest_bar = bars[-1] if bars else {}
            candle_close = _coerce_float(latest_bar.get("close")) if isinstance(latest_bar, Mapping) else None
            if candle_close is not None and candle_close > 0:
                ltp = candle_close
                source = "ws_candle_fallback"
                self._logger.info(
                    "SPOT_SNAPSHOT_FROM_CANDLE_FALLBACK symbol=%s ltp=%s",
                    canonical,
                    round(candle_close, 4),
                    extra={
                        "event": "SPOT_SNAPSHOT_FROM_CANDLE_FALLBACK",
                        "symbol": canonical,
                        "ltp": candle_close,
                        "source": source,
                    },
                )
        latest = bars[-1] if bars else {}
        latest_candle_provisional = bool(latest.get("provisional")) if isinstance(latest, Mapping) else False
        latest_candle_synthetic = bool(latest.get("synthetic")) if isinstance(latest, Mapping) else False
        ohlc_valid = bool(bars)
        depth_available = isinstance(tick.get("depth"), Mapping) and bool(tick.get("depth"))
        bid_missing = bool(tick.get("bid_missing")) or bid is None or bid <= 0
        ask_missing = bool(tick.get("ask_missing")) or ask is None or ask <= 0
        bid_ask_source = str(
            tick.get("bid_ask_source") or ("market_depth" if depth_available else "missing")
        ).lower()
        source_ok = bid_ask_source in {"market_depth", "quote", "rest_quote", "ws_full"}
        quote_source = str(tick.get("source") or "").lower()
        rest_quote_source = quote_source in {"rest", "rest_quote", "quote"}
        tradable_quote = (
            (not bid_missing and not ask_missing)
            and source_ok
            and (self._quote_api_available or not rest_quote_source)
        )
        snapshot = MarketSnapshot(
            symbol=str(symbol),
            canonical_symbol=canonical,
            ltp=ltp,
            bid=bid,
            ask=ask,
            mid=mid,
            tick_age_s=age,
            source=source,
            real_ticks_last_60s=real_ticks,
            latest_candle_provisional=latest_candle_provisional,
            latest_candle_synthetic=latest_candle_synthetic,
            ohlc_valid=ohlc_valid,
            depth_available=depth_available,
            bid_missing=bid_missing,
            ask_missing=ask_missing,
            bid_ask_source=bid_ask_source,
            tradable_quote=tradable_quote,
        )
        age_value = snapshot.tick_age_s
        stale_threshold = self._ltp_stale_threshold_for_symbol(canonical)
        if age_value is not None and age_value > stale_threshold:
            self._logger.debug(
                "MARKET_SNAPSHOT_STALE symbol=%s age=%s",
                canonical,
                round(age_value, 3),
            )
        else:
            self._logger.debug(
                "MARKET_SNAPSHOT_READY symbol=%s ltp=%s source=%s age=%s",
                canonical,
                snapshot.ltp,
                snapshot.source,
                round(age_value, 3) if age_value is not None else None,
            )
        return snapshot

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
        normalized = self._canonical_symbol(symbol)
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
        with self._lock:
            bars = list(self._ohlc.get(bar_symbol, ()))
        if limit is not None and limit >= 0:
            bars = bars[-limit:]
        return bars

    def get_ohlc(self, symbol: str) -> list[dict[str, Any]]:
        """Return finalized candles for *symbol*. Args: symbol. Returns: list of candles. Raises: None."""
        return self.get_ohlc_bars(symbol)

    @property
    def market_data(self) -> dict[str, Any]:
        """Return derived market data snapshots such as OHLC bars."""

        snapshot: dict[str, Any] = {}
        with self._lock:
            ohlc_snapshot = {symbol: list(bars) for symbol, bars in self._ohlc.items()}
        for symbol, bars in ohlc_snapshot.items():
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

        raw_symbol = str(symbol or "").strip()
        if raw_symbol.isdigit():
            return [int(raw_symbol)]
        normalized = self._canonical_symbol(raw_symbol)
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
        normalized_key: str | int = key
        if isinstance(key, str) and not key.strip().isdigit():
            normalized_key = self._canonical_symbol(key)
        return self._broker_quote(broker, normalized_key)

    def _broker_quote(self, broker: Any, broker_symbol: str | int) -> dict[str, Any] | None:
        """Fetch quote via broker.quote/get_quote variants. Args: broker + symbol. Returns: quote dict or None. Raises: none."""
        try:
            aliases: list[str] = []
            if isinstance(broker_symbol, int):
                aliases.append(str(int(broker_symbol)))
            else:
                aliases.extend(
                    [
                        str(broker_symbol),
                        str(broker_symbol).upper(),
                        str(broker_symbol).split(":", 1)[-1],
                    ]
                )
            quote_fn = getattr(broker, "quote", None)
            if callable(quote_fn):
                raw_quote = quote_fn([broker_symbol])  # type: ignore[arg-type]
                if isinstance(raw_quote, Mapping):
                    for alias in aliases:
                        payload = raw_quote.get(alias)
                        if isinstance(payload, Mapping):
                            self._mark_quote_api_status(available=True)
                            return dict(payload)
                    for payload in raw_quote.values():
                        if isinstance(payload, Mapping):
                            self._mark_quote_api_status(available=True)
                            return dict(payload)
            get_quote_fn = getattr(broker, "get_quote", None)
            if callable(get_quote_fn):
                for arg in ([broker_symbol], broker_symbol):
                    raw_quote = get_quote_fn(arg)  # type: ignore[misc]
                    if isinstance(raw_quote, Mapping):
                        for alias in aliases:
                            payload = raw_quote.get(alias)
                            if isinstance(payload, Mapping):
                                self._mark_quote_api_status(available=True)
                                return dict(payload)
                        if any(key in raw_quote for key in ("ltp", "last_price", "bid", "ask")):
                            self._mark_quote_api_status(available=True)
                            return dict(raw_quote)
                        for payload in raw_quote.values():
                            if isinstance(payload, Mapping):
                                self._mark_quote_api_status(available=True)
                                return dict(payload)
            quote_any_fn = getattr(broker, "quote_any", None)
            if callable(quote_any_fn):
                raw_any = quote_any_fn([broker_symbol])  # type: ignore[arg-type]
                if isinstance(raw_any, Mapping):
                    for alias in aliases:
                        payload = raw_any.get(alias)
                        if isinstance(payload, Mapping):
                            self._mark_quote_api_status(available=True)
                            return dict(payload)
                    for payload in raw_any.values():
                        if isinstance(payload, Mapping):
                            self._mark_quote_api_status(available=True)
                            return dict(payload)
            if isinstance(broker_symbol, int):
                get_quote_by_token = getattr(broker, "get_quote_by_token", None)
                if callable(get_quote_by_token):
                    raw_token_quote = get_quote_by_token(int(broker_symbol))
                    if isinstance(raw_token_quote, Mapping):
                        self._mark_quote_api_status(available=True)
                        return dict(raw_token_quote)
        except Exception as exc:  # noqa: BLE001
            if self._is_quote_access_denied(exc):
                self._mark_quote_api_status(available=False, error="access_denied")
            self._logger.debug(
                "Failure in _broker_quote: %s",
                exc,
                extra={"event": "mdm_quote_fetch_failed", "key": broker_symbol},
            )
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
            self._logger.info(
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
            # Normalize through the standard REST tick pipeline so all
            # freshness markers (_last_tick_time, _last_tick_wallclock,
            # _last_quote_ts_ms, _latest_ticks) are updated consistently.
            canonical_symbol = self._canonical_symbol(symbol)
            prepared = self._prepare_rest_tick(quote, source="rest")
            with self._lock:
                prev = self._latest_ticks.get(canonical_symbol)
            normalized_tick = self._normalize_tick(canonical_symbol, prepared, prev)
            if normalized_tick is not None:
                if mid is not None:
                    with self._lock:
                        self._last_mid[canonical_symbol] = (mid, now_ms)
                self._emit_tick(canonical_symbol, normalized_tick, source="rest")
                return normalized_tick
            # Fallback: raw cache update when normalization yields no ltp
            with self._lock:
                self._latest_ticks[canonical_symbol] = quote
                self._last_quote_ts_ms[canonical_symbol] = now_ms
                self._last_tick_time[canonical_symbol] = time.time()
                if mid is not None:
                    self._last_mid[canonical_symbol] = (mid, now_ms)
            return quote
        self._logger.error(
            "refresh_quote_failed_all_candidates",
            extra={
                "symbol": symbol,
                "candidates": candidates,
                "trace_id": trace_id,
            },
            exc_info=True,
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

    def quote_age_ms(self, symbol: str | int) -> int:
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
            resolved_symbol: str | None
            if isinstance(symbol, int):
                with self._lock:
                    resolved_symbol = self._symbol_by_token.get(int(symbol))
            elif str(symbol).strip().isdigit():
                with self._lock:
                    resolved_symbol = self._symbol_by_token.get(int(str(symbol).strip()))
            else:
                resolved_symbol = self._canonical_symbol(str(symbol))
            if not resolved_symbol:
                return 1_000_000_000
            with self._lock:
                ts_ms = self._last_quote_ts_ms.get(resolved_symbol)
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
        symbol = self._canonical_symbol(symbol)
        try:
            quote = self._broker.get_quote(symbol)
            if not quote or not isinstance(quote, dict):
                self._log_poll_error(symbol, "empty_quote")
                return
            quote = self._prepare_rest_tick(quote, source="rest")
            with self._lock:
                previous = self._latest_ticks.get(symbol)
            normalized = self._normalize_tick(symbol, quote, previous)
            if normalized is None:
                self._log_poll_error(symbol, "quote_normalization_failed")
                return
            normalized_live = self.normalize_live_tick(normalized, source="poll")
            if normalized_live is None:
                self._log_poll_error(symbol, "quote_normalization_failed")
                return
            ws_age = self._tick_age_seconds(symbol)
            stale_threshold = (
                self._polling_policy.context_stale_seconds
                if self._is_context_symbol(symbol)
                else self._polling_policy.option_stale_seconds
            )
            if ws_age is not None and ws_age <= stale_threshold and self._last_tick_source.get(symbol) == "ws":
                self._logger.debug(
                    "poll_tick_skipped_fresh_ws symbol=%s ws_age=%.3f threshold=%.3f",
                    symbol,
                    ws_age,
                    stale_threshold,
                    extra={
                        "event": "poll_tick_skipped_fresh_ws",
                        "symbol": symbol,
                        "ws_age": float(ws_age),
                        "threshold": float(stale_threshold),
                    },
                )
                return
            self._poll_fallback_count = int(getattr(self, "_poll_fallback_count", 0)) + 1
            self._ingest_normalized_tick(normalized_live)
            self._process_poll_quote(symbol, normalized)
            self._logger.info(
                "POLL_TICK_INGESTED symbol=%s ltp=%s",
                symbol,
                normalized_live.get("ltp"),
                extra={"event": "POLL_TICK_INGESTED", "symbol": symbol},
            )
        except Exception as exc:
            self._log_poll_error(symbol, exc)

    def _log_poll_error(self, symbol: str, err: Exception | str) -> None:
        """Throttle poll error logs. Args: symbol/err. Returns: none. Raises: none."""
        now = time.monotonic()
        key = str(symbol or "UNKNOWN")
        last = self._poll_error_last_log.get(key, 0.0)
        if now - last < self._polling_policy.error_log_throttle_seconds:
            return
        self._poll_error_last_log[key] = now
        self._logger.warning(
            "POLL_QUOTE_FAILED symbol=%s err=%s",
            key,
            err,
            extra={"event": "POLL_QUOTE_FAILED", "symbol": key, "error": str(err)},
        )

    def _is_context_symbol(self, symbol: str) -> bool:
        """Return whether symbol is context feed. Args: symbol; Returns: bool; Raises: none."""
        s = str(symbol or "").upper()
        return s == "NSE:NIFTY" or (s.startswith("NFO:NIFTY") and s.endswith("FUT"))

    def _is_option_symbol(self, symbol: str) -> bool:
        """Return whether symbol is NIFTY option. Args: symbol; Returns: bool; Raises: none."""
        s = str(symbol or "").upper()
        return s.startswith("NFO:NIFTY") and (s.endswith("CE") or s.endswith("PE"))

    def _tick_age_seconds(self, symbol: str, now: datetime) -> float | None:
        """Return tick age in seconds. Args: symbol, now; Returns: age or none; Raises: none."""
        ts = self._last_tick_time.get(symbol)
        if ts is None:
            return None
        if isinstance(ts, datetime):
            ts_dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            ts_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return max(0.0, (now - ts_dt).total_seconds())

    def _should_poll_symbol(self, symbol: str, now: datetime) -> bool:
        """Decide REST poll fallback need. Args: symbol, now; Returns: bool; Raises: none."""
        sym = self._canonical_symbol(symbol)
        with self._lock:
            active = set(self._active_subscribed_symbols) or set(self._tracked_symbols)
            if active and sym not in active:
                return False
            if not self._is_ws_connected():
                return True
            age = self._tick_age_seconds(sym, now)
        if age is None:
            return True
        threshold = (
            self._polling_policy.context_stale_seconds
            if self._is_context_symbol(sym)
            else self._polling_policy.option_stale_seconds
        )
        return age >= threshold

    def _poll_candidates(self, now: datetime) -> list[str]:
        """Build stale-only poll candidates. Args: now. Returns: symbols. Raises: none."""
        with self._lock:
            active = set(self._active_subscribed_symbols) | set(self._tracked_symbols)
        ordered = sorted(sym for sym in active if sym)
        stale: list[str] = []
        skipped_fresh = 0
        for sym in ordered:
            if self._should_poll_symbol(sym, now):
                stale.append(sym)
            else:
                skipped_fresh += 1
        limit = max(0, int(self._polling_policy.max_symbols))
        if limit > 0:
            stale = stale[:limit]
        self._logger.info(
            "POLL_CANDIDATES total=%d stale=%d skipped_fresh=%d",
            len(ordered),
            len(stale),
            skipped_fresh,
            extra={"event": "POLL_CANDIDATES"},
        )
        return stale

    def _is_ws_fresh_enough(self, symbol: str, now: datetime) -> bool:
        """Return whether websocket tick freshness is sufficient. Args: symbol, now. Returns: bool. Raises: none."""
        ts = self._last_tick_time.get(self._canonical_symbol(symbol))
        if ts is None:
            return False
        tick_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        age = (now - tick_dt).total_seconds()
        threshold = (
            self._polling_policy.context_stale_seconds
            if self._is_context_symbol(symbol)
            else self._polling_policy.option_stale_seconds
        )
        return age < threshold

    def _is_true_market_data_starvation(self, now: datetime) -> bool:
        """Return starvation status after WS/options freshness checks. Args: now. Returns: bool. Raises: none."""
        is_open_fn = getattr(self, "_is_market_open_now", None)
        if callable(is_open_fn) and not bool(is_open_fn()):
            return False
        grace_until = getattr(self, "_ws_reconnect_grace_until", None)
        if isinstance(grace_until, datetime) and now < grace_until:
            return False
        spot_age = self._tick_age_seconds("NSE:NIFTY", now)
        if spot_age is not None and spot_age < self._polling_policy.context_stale_seconds:
            return False
        recent_live_symbols = 0
        for sym in list(getattr(self, "_last_tick_time", {}).keys()):
            age = self._tick_age_seconds(sym, now)
            if age is not None and age < self._polling_policy.option_stale_seconds:
                recent_live_symbols += 1
        if recent_live_symbols > 0:
            return False
        return True

    def normalize_live_tick(
        self, raw: Mapping[str, Any], source: str = "ws"
    ) -> dict[str, Any] | None:
        """Normalize live tick. Args: raw/source. Returns: canonical tick or none. Raises: none."""
        token_raw = raw.get("token") or raw.get("instrument_token")
        token: int | None = None
        if token_raw is not None:
            with suppress(Exception):
                token = int(token_raw)
        symbol_raw = raw.get("symbol")
        symbol = str(symbol_raw) if symbol_raw else ""
        if not symbol and token is not None:
            with self._lock:
                symbol = str(self._symbol_by_token.get(token) or self._token_to_symbol.get(token) or "")
        if not symbol:
            return None
        ltp_raw = raw.get("ltp") or raw.get("last_price") or raw.get("price")
        with suppress(Exception):
            ltp = float(ltp_raw)
            if ltp <= 0:
                return None
            ts = pd.to_datetime(raw.get("timestamp") or raw.get("exchange_timestamp") or datetime.now(timezone.utc), utc=True, errors="coerce")
            if pd.isna(ts):
                ts = pd.Timestamp.utcnow()
            ts_py = ts.to_pydatetime().astimezone(timezone.utc)
            return {
                "symbol": self._canonical_symbol(symbol),
                "token": token,
                "ltp": ltp,
                "bid": _coerce_float(raw.get("bid")),
                "ask": _coerce_float(raw.get("ask")),
                "volume": _coerce_int(raw.get("volume") or raw.get("volume_traded") or raw.get("volume_traded_today")),
                "oi": _coerce_int(raw.get("oi")),
                "timestamp": ts_py,
                "source": "poll" if str(source).lower() == "poll" else "ws",
            }
        return None

    def _ltp_stale_threshold_for_symbol(self, symbol: str) -> float:
        """Return per-symbol stale threshold. Args: symbol. Returns: seconds. Raises: none."""
        market_open = get_market_state() == MarketState.OPEN
        return float(stale_threshold_for_symbol(symbol, market_open))

    def _request_spot_resubscribe_if_due(self, symbol: str, reason: str) -> None:
        """Request throttled spot resubscription when a token is available. Args: symbol/reason. Returns: none. Raises: none."""

        now = time.time()
        if now - self._last_spot_resubscribe_attempt < 120.0:
            return
        self._last_spot_resubscribe_attempt = now
        token = int(self._symbol_to_token.get(symbol) or self._token_by_symbol.get(symbol) or 0)
        if token <= 0:
            log_throttled(
                self._logger,
                key=f"spot_resubscribe_token_missing:{symbol}",
                msg="SPOT_RESUBSCRIBE_SKIPPED symbol=%s reason=token_missing",
                args=(symbol,),
                interval_sec=60.0,
                level=logging.WARNING,
                extra={
                    "event": "SPOT_RESUBSCRIBE_SKIPPED",
                    "symbol": symbol,
                    "reason": "token_missing",
                },
            )
            return
        self._logger.info(
            "WS_RESUBSCRIBE_REQUESTED symbol=%s reason=%s",
            symbol,
            reason,
            extra={"event": "WS_RESUBSCRIBE_REQUESTED", "symbol": symbol, "reason": reason},
        )
        self.request_token_subscription(token, symbol=symbol)

    def _monitor_spot_ws_health(self) -> None:
        """Monitor NSE spot tick freshness and trigger throttled resubscribe. Args: none. Returns: none. Raises: none."""
        if get_market_state() != MarketState.OPEN:
            return
        symbol = "NSE:NIFTY"
        if not self.symbol_has_tick(symbol):
            log_throttled(
                self._logger,
                key="spot_ws_first_tick_missing",
                msg="MDM_WS_FIRST_TICK_MISSING symbol=%s reason=no_tick_received connected=%s connect_task_alive=%s ws_tokens=%s",
                args=(symbol, self.ws_status_snapshot().get("connected"), self.ws_status_snapshot().get("connect_task_alive"), self.ws_status_snapshot().get("tokens")),
                interval_sec=60.0,
                level=logging.WARNING,
                extra={"event": "MDM_WS_FIRST_TICK_MISSING", "symbol": symbol, "reason": "no_tick_received", "connected": self.ws_status_snapshot().get("connected"), "connect_task_alive": self.ws_status_snapshot().get("connect_task_alive"), "ws_tokens": self.ws_status_snapshot().get("tokens")},
            )
            self._request_spot_resubscribe_if_due(symbol, reason="first_tick_missing")
            return
        age_ms = self.symbol_data_age_ms_or_none(symbol)
        if age_ms is None:
            return
        threshold_ms = self._ltp_stale_threshold_for_symbol(symbol) * 1000.0
        self._logger.debug(
            "SPOT_WS_HEALTH_CHECK symbol=%s age=%.2fs threshold=%.2fs",
            symbol,
            age_ms / 1000.0,
            threshold_ms / 1000.0,
            extra={"event": "SPOT_WS_HEALTH_CHECK", "symbol": symbol, "age_s": round(age_ms / 1000.0, 3), "threshold_s": round(threshold_ms / 1000.0, 3)},
        )
        if age_ms <= threshold_ms:
            return
        now = time.time()
        if now - self._last_spot_stale_log >= 60.0:
            self._last_spot_stale_log = now
            self._logger.warning(
                "MDM_WS_TICK_STALE symbol=%s age=%.2fs",
                symbol,
                age_ms / 1000.0,
                extra={"event": "MDM_WS_TICK_STALE", "symbol": symbol, "age_s": round(age_ms / 1000.0, 3)},
            )
        self._request_spot_resubscribe_if_due(symbol, reason="stale_index_tick")


    def _has_recent_rest_ticks(self) -> bool:
        cutoff = time.time() - max(self._rest_poll_interval * 2.0, 5.0)
        with self._lock:
            for symbol, ts in self._last_tick_wallclock.items():
                source = self._last_tick_source.get(symbol)
                if source in {"rest", "poll"} and ts >= cutoff:
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
        try:
            symbol = self._canonical_symbol(symbol)
            changed = self.request_symbol_subscription(symbol)
            if changed:
                token = self._symbol_to_token.get(symbol)
                if token:
                    if symbol == "NSE:NIFTY":
                        self._logger.info(
                            "SPOT_TOKEN_RESOLVED symbol=%s token=%s",
                            symbol,
                            token,
                            extra={
                                "event": "SPOT_TOKEN_RESOLVED",
                                "symbol": symbol,
                                "token": token,
                            },
                        )
                    self._logger.info(
                        "WS_SUBSCRIBE_REQUESTED symbol=%s token=%s",
                        symbol,
                        token,
                        extra={
                            "event": "WS_SUBSCRIBE_REQUESTED",
                            "symbol": symbol,
                            "token": token,
                        },
                    )
                    if not hasattr(self._ws, "is_connected") or self._is_ws_connected():
                        self._dispatched_subscriptions.add(int(token))
                        self._logger.info(
                            "WS_SUBSCRIBE_DISPATCHED symbol=%s token=%s",
                            symbol,
                            token,
                            extra={
                                "event": "WS_SUBSCRIBE_DISPATCHED",
                                "symbol": symbol,
                                "token": token,
                            },
                        )
                    else:
                        self._pending_subscriptions.add(int(token))
                        self._pending_subscription_tokens.add(int(token))
                        self._logger.info(
                            "WS_SUBSCRIBE_QUEUED symbol=%s token=%s",
                            symbol,
                            token,
                            extra={
                                "event": "WS_SUBSCRIBE_QUEUED",
                                "symbol": symbol,
                                "token": token,
                            },
                        )
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
        try:
            self.request_symbol_unsubscription(symbol)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "Unsubscribe failed",
                extra={"symbol": symbol, "error": str(exc)},
            )

    def _resolve_token(self, symbol: str) -> int | None:
        symbol = self._canonical_symbol(symbol)
        # 1. Check local cache
        token = self._token_by_symbol.get(symbol)
        if token:
            return token

        # 2. Robust fallback: Ask broker for this specific instrument
        try:
            # We strip exchange prefixes if present for the search
            clean_symbol = symbol.split(":")[-1]
            # Route to the correct exchange: derivatives end in FUT/CE/PE → NFO,
            # everything else → NSE.  Previously the call had no exchange argument
            # which defaulted to NSE, so all NFO instruments silently returned None.
            _suffix = clean_symbol.upper()
            _exchange = "NFO" if any(_suffix.endswith(s) for s in ("FUT", "CE", "PE")) else "NSE"
            instruments = self._broker.instruments(_exchange)
            for ins in instruments:
                if ins["tradingsymbol"] == clean_symbol:
                    t = int(ins["instrument_token"])
                    self.register_symbol(symbol, t) # Cache it
                    return t
        except Exception as e:
            self._logger.error(f"Failed live token resolution for {symbol}: {e}")
        if symbol == "NSE:NIFTY":
            self._logger.error(
                "SPOT_TOKEN_RESOLUTION_FAILED symbol=%s reason=%s",
                symbol,
                "token_not_found",
                extra={
                    "event": "SPOT_TOKEN_RESOLUTION_FAILED",
                    "symbol": symbol,
                    "reason": "token_not_found",
                },
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

        # 4. Preserve missing bid/ask (do not fabricate from LTP)
        if bid is None:
            buy_levels = depth.get("buy")
            if isinstance(buy_levels, list) and not buy_levels:
                self._logger.debug(
                    "Depth present but buy side empty", extra={"symbol": symbol}
                )

        if ask is None:
            sell_levels = depth.get("sell")
            if isinstance(sell_levels, list) and not sell_levels:
                self._logger.debug(
                    "Depth present but sell side empty", extra={"symbol": symbol}
                )

        bid_missing = bid is None or float(bid) <= 0
        ask_missing = ask is None or float(ask) <= 0
        bid_ask_source = "market_depth" if depth else "missing"
        source = tick.get("source")
        if isinstance(source, str) and source:
            source_lower = source.lower()
            if source_lower in {"quote", "rest_quote", "ws_full"} and not depth:
                bid_ask_source = source_lower
        source_ok = bid_ask_source in {"market_depth", "quote", "rest_quote", "ws_full"}
        source_lower = str(source or "").lower()
        rest_quote_source = source_lower in {"rest", "rest_quote", "quote"}
        tradable_quote = (
            (not bid_missing and not ask_missing)
            and source_ok
            and (self._quote_api_available or not rest_quote_source)
        )

        timestamp = self._coerce_timestamp(tick)

        normalized = {
            "symbol": symbol,
            "ltp": float(ltp),
            "last_price": float(ltp),
            "bid": float(bid) if not bid_missing else None,
            "ask": float(ask) if not ask_missing else None,
            "timestamp": timestamp,
            "received_at": self._parse_wallclock(
                tick.get("received_at") or tick.get("wallclock")
            )
            or time.time(),
            "depth": depth,
            "bid_missing": bid_missing,
            "ask_missing": ask_missing,
            "bid_ask_source": bid_ask_source,
            "tradable_quote": tradable_quote,
            "ltq": tick.get("last_quantity"),
            "oi": self._coerce_float(tick, "oi", "open_interest"),
        }
        if isinstance(source, str) and source:
            normalized["source"] = source
            if tradable_quote and not depth:
                normalized["bid_ask_source"] = source.lower()
        broker_timestamp = tick.get("broker_timestamp")
        if broker_timestamp is not None:
            normalized["broker_timestamp"] = broker_timestamp

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
        """Return best depth price for side='buy' or side='sell' without fabricating prices."""
        if not isinstance(depth, Mapping):
            return None
        entries = depth.get(side)
        if not isinstance(entries, Iterable) or isinstance(
            entries, (str, bytes, dict)
        ):
            return None
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            price = entry.get("price")
            if price is None:
                continue
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    @staticmethod
    def _coerce_timestamp(tick: dict[str, Any]) -> float:
        value = (
            tick.get("received_at")
            or tick.get("wallclock")
            or tick.get("timestamp")
            or tick.get("ts")
            or tick.get("ts_ms")
        )
        parsed = MarketDataManager._parse_wallclock(value)
        return parsed if parsed is not None else time.time()

    @staticmethod
    def _parse_wallclock(value: Any) -> float | None:
        """Parse timestamp-like values into epoch seconds."""
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        if isinstance(value, (int, float)):
            val = float(value)
            if val > 1_000_000_000_000:
                val /= 1000.0
            return val if val > 0 else None
        if isinstance(value, str) and value.strip():
            with suppress(ValueError):
                as_float = float(value.strip())
                return MarketDataManager._parse_wallclock(as_float)
            with suppress(ValueError):
                dt_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return MarketDataManager._parse_wallclock(dt_value)
        return None

    def _tick_wallclock(self, tick: Mapping[str, Any]) -> float | None:
        """Return wall-clock for tick freshness; prefer received_at."""
        for key in ("received_at", "wallclock", "timestamp"):
            parsed = self._parse_wallclock(tick.get(key))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _prepare_rest_tick(tick: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        payload = dict(tick)
        broker_timestamp = payload.get("broker_timestamp")
        if broker_timestamp is None:
            broker_timestamp = (
                payload.get("timestamp") or payload.get("ts") or payload.get("ts_ms")
            )
        if broker_timestamp is not None:
            payload["broker_timestamp"] = broker_timestamp
        observed_at = payload.pop("_local_timestamp", None)
        payload["source"] = source
        if isinstance(observed_at, (int, float)):
            payload["timestamp"] = float(observed_at)
            payload["received_at"] = float(observed_at)
        else:
            payload["timestamp"] = time.time()
            payload["received_at"] = time.time()
        return payload

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


    @staticmethod
    def normalize_history_row(symbol: str, row: Any, source: str = "historical") -> dict[str, Any] | None:
        """Normalize broker/DataHub OHLC row. Args: symbol/row/source. Returns: canonical bar or None. Raises: None."""
        return normalize_history_row(symbol, row, source=source)

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
                except Exception as e:
                    self._logger.exception("Unhandled exception", exc_info=True)
                    raise

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
                except Exception as e:
                    self._logger.exception("Unhandled exception", exc_info=True)
                    raise

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
                async with self._history_lock:
                    now_mono = time.monotonic()
                    wait = self._history_min_interval_sec - (
                        now_mono - self._last_history_request_ts
                    )
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_history_request_ts = time.monotonic()
                    try:
                        data = await asyncio.to_thread(
                            fetcher, token, from_date, to_date, interval
                        )
                    except Exception as exc:
                        err = str(exc)
                        if "Rate limit exceeded" in err or "zerodha.historical" in err:
                            self._logger.warning(
                                "HISTORY_RATE_LIMIT_BACKOFF symbol=%s sleep=2.5",
                                symbol,
                                extra={"event": "HISTORY_RATE_LIMIT_BACKOFF", "symbol": symbol},
                            )
                            await asyncio.sleep(2.5)
                            self._last_history_request_ts = time.monotonic()
                            data = await asyncio.to_thread(
                                fetcher, token, from_date, to_date, interval
                            )
                        else:
                            raise
                raw_rows = data or []
                normalized = [
                    bar
                    for bar in (
                        self.normalize_history_row(symbol, row, source="historical")
                        for row in raw_rows
                    )
                    if bar is not None
                ]

                return normalized

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
