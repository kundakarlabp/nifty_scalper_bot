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
from nifty_scalper_bot.data.websocket.manager import ConnectionState, WebSocketManager
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.env import get_str
from nifty_scalper_bot.utils.logging import get_logger, get_tracer_logger
from nifty_scalper_bot.utils.metrics import Counter

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

# ...[snip, no change needed above this line]...

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
        # Resolver may be injected during construction or attached later by app.py.
        self._resolver = resolver

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
        self._tracked_symbols: set[str] = set()
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
            "MDM_POLL_MAX_SYMBOLS", default=5, minimum=1
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
        self._tick_stale_threshold_ms = self._parse_int_env(
            "TICK_STALE_MS", default=2_000, minimum=0
        )
        if self._rest_poll_enabled:
            self._tracked_symbols.add("NIFTY")

        try:
            settings = get_settings()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketDataManager.__init__: %s",
                exc,
                extra={"event": "mdm_settings_load_failed"},
            )
            settings = None
        if settings is not None and self._resolver is None:
            try:
                self._resolver = getattr(settings, "resolver", None)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarketDataManager.__init__: %s",
                    exc,
                    extra={"event": "mdm_settings_resolver_invalid"},
                )
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
        self._m_invalid_ticks = Counter("mdm_invalid_ticks_total", "Invalid ticks dropped")
        self._m_stale_ticks = Counter("mdm_stale_ticks_total", "Stale ticks dropped")

# ...existing unchanged...

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        # PATCH: Validate tick shape from PollingStreamer
        if not isinstance(tick, dict):
            self._logger.error("[POLL-ERR] Tick is not a dict: %s", type(tick))
            with suppress(Exception):
                self._m_invalid_ticks.inc()
            return
        if (
            "instrument_token" not in tick
            and "token" not in tick
        ):
            self._logger.error("[POLL-ERR] Tick missing token fields: %s", tick)
            with suppress(Exception):
                self._m_invalid_ticks.inc()
            return
        raw_token = tick.get("instrument_token")
# ...rest of method unchanged, but... escalate stale tick error...
                if age_ms > stale_threshold:
                    self._logger.error(
                        "[POLL-ERR] Stale tick dropped",
                        extra={
                            "symbol": symbol,
                            "age_ms": round(age_ms, 2),
                            "threshold_ms": stale_threshold,
                        },
                    )
                    with suppress(Exception):
                        self._m_stale_ticks.inc()
                    return
# ...
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

        # Calculate staleness readiness
        max_tick_age = max((age for age in last_tick_age.values() if age is not None), default=None)
        ready = max_tick_age is not None and max_tick_age < (self._tick_stale_threshold_ms / 1000.0)

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
            "ready": ready,
            "max_tick_age_s": max_tick_age,
        }
