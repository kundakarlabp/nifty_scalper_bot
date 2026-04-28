"""Canonical SSOT cache for ticks, positions, orders, and broker facades."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

from nifty_scalper_bot.storage.hub_store import HubStore
from nifty_scalper_bot.utils.options_math import black_scholes_greeks, implied_volatility
from nifty_scalper_bot.utils.symbols import canonical

if TYPE_CHECKING:
    from nifty_scalper_bot.core.message_bus import Message

LOGGER = logging.getLogger(__name__)

Tick = Dict[str, Any]
TickListener = Callable[[Tick], None]
OrderListener = Callable[[dict[str, Any]], None]


class _CompletedAwaitable:
    """Provide await-compatibility for sync-first ingestion APIs."""

    def __await__(self):
        if False:
            yield None
        return None


class _EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event_name, ()))
        for handler in handlers:
            handler(payload)


class TickBus:
    """Backward-compatible tick event bus."""

    def __init__(self) -> None:
        self._event_bus = _EventBus()

    def subscribe(self, handler: TickListener) -> None:
        self._event_bus.subscribe("tick", handler)

    def publish(self, tick: dict[str, Any]) -> None:
        self._event_bus.publish("tick", tick)

    def subscribe_event(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._event_bus.subscribe(event_name, handler)

    def publish_event(self, event_name: str, payload: dict[str, Any]) -> None:
        self._event_bus.publish(event_name, payload)


_ORDER_STATE_MACHINE: dict[str, set[str]] = {
    "": {"pending", "submitted", "open", "partial_filled", "filled", "cancelled", "rejected", "expired"},
    "pending": {"submitted", "open", "partial_filled", "cancelled", "rejected", "expired"},
    "open": {"submitted", "partial_filled", "filled", "cancelled", "rejected", "expired"},
    "submitted": {"partial_filled", "filled", "cancelled", "rejected", "expired"},
    "partial_filled": {"partial_filled", "filled", "cancelled", "rejected", "expired"},
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
    "expired": set(),
}

_MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_OPTION_RE = re.compile(
    r"^(?:NFO:)?(?P<base>[A-Z]+)(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<strike>\d+)(?P<side>CE|PE)$"
)


class DataHub:
    """Market data single source of truth."""

    def __init__(
        self,
        market_data_manager: Any,
        instrument_resolver: Any = None,
        options_only: bool = False,
        **kwargs,
    ):
        self._mdm = market_data_manager
        self._resolver = instrument_resolver
        self._options_only = bool(options_only)
        self._store: HubStore | None = kwargs.get("store")
        self._event_bus = kwargs.get("event_bus")
        self._message_bus = kwargs.get("message_bus") or kwargs.get("event_bus")
        self._lock = threading.RLock()

        self._ticks: Dict[int, Tick] = {}
        self._quotes: Dict[str, Tick] = {}
        self._positions: Dict[Any, dict[str, Any]] = {}
        self._orders: Dict[str, dict[str, Any]] = {}
        self._token_by_symbol: Dict[str, int] = {}
        self._symbol_by_token: Dict[int, str] = {}
        self._tick_subscribers: Dict[str, set[TickListener]] = defaultdict(set)
        self._order_subscribers: list[OrderListener] = []
        self._mdm_subscribed_symbols: set[str] = set()
        self._defer_live_symbol_subscriptions = bool(
            kwargs.get("defer_live_symbol_subscriptions", True)
        )
        self._pending_live_symbols: set[str] = set()

        self._last_ts: Dict[str, float] = {}
        self._last_arrival: Dict[str, float] = {}
        self._last_ws_arrival: Dict[str, float] = {}
        self._last_poll_arrival: Dict[str, float] = {}
        self._last_global_ws_arrival = 0.0
        self._last_arrival_mono: Dict[str, float] = {}
        self._stale_candidates: Dict[str, int] = defaultdict(int)
        self._last_listener_error_log_ts = 0.0
        self._listener_error_streak = 0
        self._history_cache: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        self._iv_cache: Dict[str, float] = {}
        self._oi_cache: Dict[str, float] = {}
        self._greeks_cache: Dict[str, dict[str, float]] = {}

        self._clock = kwargs.get("clock") or time.time
        self._now = kwargs.get("clock") or time.time
        self._monotonic = time.monotonic
        self._start_mono = self._monotonic()
        self._poll_block_ms = 800.0
        self._stale_tick_max_age_ms = float(os.getenv("STALE_TICK_MAX_AGE_MS", "2000"))
        self._drift_budget_ms = float(os.getenv("DRIFT_BUDGET_MS", "250"))
        self._warmup_grace_s = float(os.getenv("WARMUP_GRACE_S", "3"))
        self._tick_error_log_cooldown_sec = float(os.getenv("TICK_ERROR_LOG_COOLDOWN_SEC", "30"))

        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.tick_bus = TickBus()

        self._bind_to_mdm()
        self._restore_snapshots()

    def _bind_to_mdm(self) -> None:
        attach_tick_bus = getattr(self._mdm, "attach_tick_bus", None)
        if callable(attach_tick_bus):
            try:
                attach_tick_bus(self.tick_bus)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("attach_tick_bus failed: %s", exc)
        LOGGER.info("DataHub bound to MDM (per-symbol subscription on demand)")

    def _restore_snapshots(self) -> None:
        if self._store is None:
            return
        try:
            quotes = self._store.load_snapshot("quotes")
            orders = self._store.load_snapshot("orders")
            positions = self._store.load_snapshot("positions")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DataHub restore failed: %s", exc)
            return
        with self._lock:
            if isinstance(quotes, dict):
                self._quotes = {str(k): dict(v) for k, v in quotes.items() if isinstance(v, dict)}
            if isinstance(orders, dict):
                self._orders = {str(k): dict(v) for k, v in orders.items() if isinstance(v, dict)}
            if isinstance(positions, dict):
                self._positions = {k: dict(v) for k, v in positions.items() if isinstance(v, dict)}

    def checkpoint(self) -> None:
        if self._store is None:
            return
        with self._lock:
            quotes = {k: dict(v) for k, v in self._quotes.items()}
            orders = {k: dict(v) for k, v in self._orders.items()}
            positions = {str(k): dict(v) for k, v in self._positions.items()}
        self._store.save_snapshot("quotes", quotes)
        self._store.save_snapshot("orders", orders)
        self._store.save_snapshot("positions", positions)

    def reset_warmup(self) -> None:
        self._start_mono = self._monotonic()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._main_loop is not None:
            LOGGER.warning("DataHub event loop already wired - ignoring rewire")
            return
        self._main_loop = loop
        mdm_set = getattr(self._mdm, "set_event_loop", None)
        if callable(mdm_set):
            try:
                mdm_set(loop)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("DataHub.set_event_loop -> MDM failed: %s", exc)

    def _canonical_quote_symbol(self, symbol: str) -> str:
        return canonical(str(symbol or ""))

    def _position_symbol(self, symbol: str) -> str:
        raw = str(symbol or "").strip().upper()
        if ":" in raw:
            _, raw = raw.split(":", 1)
        return raw

    def _position_allowed(self, symbol: str) -> bool:
        if not self._options_only:
            return True
        return symbol == "NIFTY" or symbol.startswith("NIFTY")

    def _position_key(self, position: dict[str, Any]) -> Any:
        token = position.get("instrument_token")
        if token is not None:
            try:
                return int(token)
            except (TypeError, ValueError):
                return str(token)
        return position["symbol"]

    def _normalize_position(self, position: dict[str, Any]) -> dict[str, Any] | None:
        symbol = self._position_symbol(position.get("symbol"))
        if not symbol or not self._position_allowed(symbol):
            return None
        try:
            qty_raw = float(position.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty_raw = 0.0
        quantity = int(abs(qty_raw))
        try:
            average_price = float(position.get("average_price", 0) or 0)
        except (TypeError, ValueError):
            average_price = 0.0
        side = "FLAT"
        if qty_raw > 0:
            side = "LONG"
        elif qty_raw < 0:
            side = "SHORT"
        normalized = dict(position)
        normalized["symbol"] = symbol
        normalized["quantity"] = quantity
        normalized["average_price"] = average_price
        normalized["side"] = side
        return normalized

    def replace_positions(self, positions: Iterable[dict]) -> None:
        new_positions: Dict[Any, dict[str, Any]] = {}
        for position in positions:
            normalized = self._normalize_position(position)
            if normalized is None:
                continue
            new_positions[self._position_key(normalized)] = normalized
        with self._lock:
            self._positions = new_positions
        self.checkpoint()
        LOGGER.debug("Positions replaced in DataHub | count=%s", len(self._positions))

    def get_positions(self) -> dict:
        with self._lock:
            return dict(self._positions)

    def positions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(position) for position in self._positions.values()]

    def update_position(self, position: dict) -> None:
        normalized = self._normalize_position(position)
        if normalized is None:
            return
        with self._lock:
            self._positions[self._position_key(normalized)] = normalized
        self.checkpoint()

    def clear_positions(self) -> None:
        with self._lock:
            self._positions = {}
        self.checkpoint()

    def _normalize_order(self, order: dict[str, Any]) -> dict[str, Any] | None:
        order_id = str(order.get("order_id") or "").strip()
        if not order_id:
            return None
        normalized = dict(order)
        normalized["order_id"] = order_id
        normalized["symbol"] = self._position_symbol(order.get("symbol"))
        normalized["side"] = str(order.get("side") or "").strip().lower() or None
        normalized["status"] = str(order.get("status") or "").strip().lower()
        try:
            normalized["status_sequence"] = int(order.get("status_sequence") or 0)
        except (TypeError, ValueError):
            normalized["status_sequence"] = 0
        for field in ("quantity", "filled_quantity"):
            try:
                normalized[field] = int(float(order.get(field, 0) or 0))
            except (TypeError, ValueError):
                normalized[field] = 0
        for field in ("price", "trigger_price"):
            raw = order.get(field)
            if raw is None or str(raw).strip() == "":
                normalized[field] = None
            else:
                try:
                    normalized[field] = float(raw)
                except (TypeError, ValueError):
                    normalized[field] = None
        timestamp = order.get("timestamp")
        try:
            ts = float(timestamp)
            normalized["ts"] = ts / 1000.0 if ts > 1e11 else ts
        except (TypeError, ValueError):
            normalized["ts"] = self._now()
        return normalized

    def ingest_order_update(self, order: dict[str, Any]) -> None:
        normalized = self._normalize_order(order)
        if normalized is None:
            return
        order_id = normalized["order_id"]
        with self._lock:
            existing = self._orders.get(order_id)
            if existing is not None:
                existing_sequence = int(existing.get("status_sequence") or 0)
                incoming_sequence = int(normalized.get("status_sequence") or 0)
                if incoming_sequence and existing_sequence and incoming_sequence < existing_sequence:
                    return
                current_status = str(existing.get("status") or "")
                next_status = str(normalized.get("status") or "")
                allowed = _ORDER_STATE_MACHINE.get(current_status, set())
                if next_status and next_status != current_status and next_status not in allowed:
                    return
            self._orders[order_id] = normalized
            subscribers = list(self._order_subscribers)
        for callback in subscribers:
            try:
                callback(dict(normalized))
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("order_listener_failed: %s", exc)
        self.checkpoint()

    def upsert_order(self, order: dict[str, Any]) -> None:
        self.ingest_order_update(order)

    def orders(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(order) for order in self._orders.values()]

    def store_quote(
        self,
        symbol: str,
        quote_data: dict[str, Any],
        source: str = "ws",
        seed: bool = False,
    ) -> None:
        payload = dict(quote_data)
        payload["symbol"] = symbol
        payload["source"] = source
        if seed:
            payload["seed"] = True
        if "timestamp" not in payload:
            payload["timestamp"] = self._now()
        self._ingest_tick_impl(payload)

    def _normalize_tick_symbol(self, tick: Tick) -> str:
        raw_symbol = str(tick.get("symbol", "")).strip()
        if not raw_symbol:
            return ""
        return self._canonical_quote_symbol(raw_symbol)

    def _timestamp_ms(self, timestamp: Any) -> float:
        if isinstance(timestamp, datetime):
            return timestamp.timestamp() * 1000.0
        if isinstance(timestamp, (int, float)):
            value = float(timestamp)
            return value * 1000.0 if value < 1e11 else value
        return self._now() * 1000.0

    def _tick_price(self, tick: dict[str, Any]) -> float | None:
        for key in ("ltp", "last_price", "price", "close"):
            value = tick.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _tick_bid(self, tick: dict[str, Any]) -> float | None:
        for key in ("bid", "best_bid"):
            value = tick.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _tick_ask(self, tick: dict[str, Any]) -> float | None:
        for key in ("ask", "best_ask"):
            value = tick.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _should_drop_stale_tick(self, symbol: str, ts_ms: float, now_ms: float, mono_now: float) -> bool:
        last_ts = self._last_ts.get(symbol, 0.0)
        if ts_ms > last_ts:
            self._stale_candidates[symbol] = 0
            return False

        server_age_ms = max(0.0, now_ms - ts_ms)
        last_mono = self._last_arrival_mono.get(symbol, self._start_mono)
        mono_age_ms = max(0.0, (mono_now - last_mono) * 1000.0)
        self._last_arrival_mono[symbol] = mono_now

        if mono_now - self._start_mono < self._warmup_grace_s:
            hard_cap_ms = max(self._stale_tick_max_age_ms * 4.0, (self._warmup_grace_s + 5.0) * 1000.0)
            if server_age_ms <= hard_cap_ms:
                self._stale_candidates[symbol] = 0
                return False

        if server_age_ms <= (self._stale_tick_max_age_ms + self._drift_budget_ms):
            self._stale_candidates[symbol] = 0
            return False
        if mono_age_ms <= (self._stale_tick_max_age_ms + self._drift_budget_ms):
            self._stale_candidates[symbol] = 0
            return False

        self._stale_candidates[symbol] += 1
        return self._stale_candidates[symbol] >= 2

    def _log_listener_failure(self, exc: Exception) -> None:
        self._listener_error_streak += 1
        now = self._now()
        if now - self._last_listener_error_log_ts < self._tick_error_log_cooldown_sec:
            return
        self._last_listener_error_log_ts = now
        LOGGER.error(
            "tick_listener_failed consecutive=%s error=%s",
            self._listener_error_streak,
            exc,
        )

    def _capture_option_metrics(self, symbol: str, tick: Tick) -> None:
        iv_raw = tick.get("implied_volatility") or tick.get("iv")
        if iv_raw is not None:
            try:
                self._iv_cache[symbol] = float(iv_raw)
            except (TypeError, ValueError):
                pass
        greeks = tick.get("greeks")
        if isinstance(greeks, dict):
            normalized_greeks = {}
            for key, value in greeks.items():
                try:
                    normalized_greeks[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if normalized_greeks:
                self._greeks_cache[symbol] = normalized_greeks
        oi = tick.get("oi") or tick.get("open_interest")
        if oi is not None:
            try:
                self._oi_cache[symbol] = float(oi)
            except (TypeError, ValueError):
                pass
        if symbol in self._iv_cache and symbol in self._greeks_cache:
            return
        parsed = self._parse_option_symbol(symbol)
        if parsed is None:
            return
        base, expiry, strike, is_call = parsed
        ltp = self._tick_price(tick)
        if ltp is None or ltp <= 0:
            return
        spot = self.get_latest_price(base)
        if spot is None or spot <= 0:
            return
        ttm_years = max((expiry.timestamp() - self._now()) / 31557600.0, 0.0)
        if ttm_years <= 0:
            return
        try:
            iv = implied_volatility(
                price=float(ltp),
                S=float(spot),
                K=float(strike),
                t=float(ttm_years),
                r=0.06,
                flag="c" if is_call else "p",
            )
        except Exception:  # noqa: BLE001
            return
        if iv is None or iv <= 0:
            return
        self._iv_cache[symbol] = float(iv)
        try:
            self._greeks_cache[symbol] = black_scholes_greeks(
                S=float(spot),
                K=float(strike),
                t=float(ttm_years),
                r=0.06,
                sigma=float(iv),
                flag="c" if is_call else "p",
            )
        except Exception:  # noqa: BLE001
            pass

    def _parse_option_symbol(
        self, symbol: str
    ) -> tuple[str, datetime, float, bool] | None:
        match = _OPTION_RE.match(symbol)
        if not match:
            return None
        year = 2000 + int(match.group("year"))
        month = _MONTH_MAP.get(match.group("month"))
        if month is None:
            return None
        expiry = datetime(year, month, 28, 15, 30, tzinfo=timezone.utc).replace(tzinfo=None)
        return (
            match.group("base"),
            expiry,
            float(match.group("strike")),
            match.group("side") == "CE",
        )

    def _ingest_tick_impl(self, tick: Tick) -> None:
        if not tick:
            return
        symbol = self._normalize_tick_symbol(tick)
        if not symbol:
            return
        token = tick.get("instrument_token")
        try:
            token = int(token) if token is not None else None
        except (TypeError, ValueError):
            token = None
        source = str(tick.get("source", "ws") or "ws").lower()
        if source == "rest":
            source = "poll"
        ts_ms = self._timestamp_ms(tick.get("timestamp", self._now()))
        now_ms = self._now() * 1000.0
        mono_now = self._monotonic()

        with self._lock:
            if self._should_drop_stale_tick(symbol, ts_ms, now_ms, mono_now):
                LOGGER.warning("stale_tick_dropped symbol=%s age_ms=%.1f", symbol, now_ms - ts_ms)
                return
            last_ts = self._last_ts.get(symbol, 0.0)
            last_arr = self._last_arrival.get(symbol, 0.0)
            last_ws = self._last_ws_arrival.get(symbol, 0.0)
            if ts_ms < last_ts:
                return
            if ts_ms == last_ts and now_ms <= last_arr:
                return
            if source == "poll" and (now_ms - last_ws) < self._poll_block_ms:
                return

            canonical_tick = dict(tick)
            canonical_tick["symbol"] = symbol
            canonical_tick["source"] = source
            canonical_tick["timestamp"] = ts_ms
            canonical_tick["arrival_time"] = now_ms
            if token is not None:
                canonical_tick["instrument_token"] = token
                self._ticks[token] = canonical_tick
                self._token_by_symbol[symbol] = token
                self._symbol_by_token[token] = symbol
            self._quotes[symbol] = canonical_tick
            self._last_ts[symbol] = ts_ms
            self._last_arrival[symbol] = now_ms
            self._last_arrival_mono[symbol] = mono_now
            if source in {"ws", "websocket", "stream"}:
                self._last_ws_arrival[symbol] = now_ms
                self._last_global_ws_arrival = now_ms
            elif source in {"poll", "rest"}:
                self._last_poll_arrival[symbol] = now_ms
            self._stale_candidates[symbol] = 0
            subscribers = list(self._tick_subscribers.get(symbol, ()))
        trace_id = str(tick.get("trace_id") or self._make_trace_id(symbol))
        canonical_tick["trace_id"] = trace_id
        LOGGER.debug(
            "DATAHUB_SYMBOL_DISPATCH symbol=%s source=%s subscribers=%d",
            symbol,
            source,
            len(subscribers),
            extra={
                "event": "DATAHUB_SYMBOL_DISPATCH",
                "symbol": symbol,
                "trace_id": trace_id,
                "source": source,
                "subscriber_count": len(subscribers),
            },
        )

        self._capture_option_metrics(symbol, canonical_tick)

        for callback in subscribers:
            try:
                callback(dict(canonical_tick))
            except Exception as exc:  # noqa: BLE001
                self._log_listener_failure(exc)
        self.checkpoint()

    def ingest_tick_sync(self, tick: Tick):
        self._ingest_tick_impl(tick)
        return _CompletedAwaitable()

    async def ingest_tick_from_bus(self, message: "Message") -> None:
        self._ingest_tick_impl(message.data)

    def ingest_tick(self, tick: Tick):
        self._ingest_tick_impl(tick)
        return _CompletedAwaitable()

    def get_quote(self, symbol: str, allow_pull: bool = True) -> Optional[Tick]:
        lookup = self._canonical_quote_symbol(symbol)
        with self._lock:
            token = self._token_by_symbol.get(lookup)
            if token is not None and token in self._ticks:
                return dict(self._ticks[token])
            tick = self._quotes.get(lookup)
            if tick is not None:
                return dict(tick)
        if allow_pull:
            return self.pull_quote(symbol) or None
        return None

    def get_tick_by_token(self, token: int) -> Optional[Tick]:
        with self._lock:
            tick = self._ticks.get(token)
            return dict(tick) if tick else None

    def subscribe(self, symbol: str, callback: Optional[TickListener] = None):
        self.subscribe_ticks(symbol, callback)

    subscribe_ticks = subscribe

    def _make_trace_id(self, symbol: str) -> str:
        """Produce a lightweight per-symbol trace id for lifecycle logs."""
        try:
            return f"{symbol}-{time.monotonic_ns()}"
        except Exception:  # noqa: BLE001
            return f"{symbol}-{int(time.time()*1e9)}"

    def _subscribe_symbol(self, symbol: str) -> None:
        trace_id = self._make_trace_id(symbol)
        if symbol.isdigit():
            LOGGER.info(
                "DATAHUB_LIVE_SUBSCRIBE symbol=%s subscribed=false reason=symbol_is_token",
                symbol,
                extra={
                    "event": "DATAHUB_LIVE_SUBSCRIBE",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "subscribed": False,
                    "reason": "symbol_is_token",
                    "mdm_delegate_called": False,
                },
            )
            return
        if symbol in self._mdm_subscribed_symbols:
            LOGGER.info(
                "DATAHUB_LIVE_SUBSCRIBE symbol=%s subscribed=true reason=already_subscribed",
                symbol,
                extra={
                    "event": "DATAHUB_LIVE_SUBSCRIBE",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "subscribed": True,
                    "reason": "already_subscribed",
                    "mdm_delegate_called": False,
                },
            )
            return
        mdm_sub = getattr(self._mdm, "subscribe", None)
        if callable(mdm_sub):
            mdm_sub(symbol, self.ingest_tick_sync)
            self._mdm_subscribed_symbols.add(symbol)
            LOGGER.info(
                "datahub_live_symbol_subscribed symbol=%s",
                symbol,
                extra={"event": "datahub_live_symbol_subscribed", "symbol": symbol},
            )
            LOGGER.info(
                "DATAHUB_LIVE_SUBSCRIBE symbol=%s subscribed=true reason=mdm_subscribed",
                symbol,
                extra={
                    "event": "DATAHUB_LIVE_SUBSCRIBE",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "subscribed": True,
                    "reason": "mdm_subscribed",
                    "mdm_delegate_called": True,
                },
            )
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=active",
                symbol,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "state": "active",
                    "deferred_mode": self._defer_live_symbol_subscriptions,
                    "callback_attached": bool(self._tick_subscribers.get(symbol)),
                    "mdm_subscribe_requested": True,
                    "success": True,
                    "pending_count": len(self._pending_live_symbols),
                    "reason": "mdm_subscribed",
                },
            )
        else:
            LOGGER.warning(
                "DATAHUB_LIVE_SUBSCRIBE symbol=%s subscribed=false reason=mdm_no_subscribe",
                symbol,
                extra={
                    "event": "DATAHUB_LIVE_SUBSCRIBE",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "subscribed": False,
                    "reason": "mdm_no_subscribe_callable",
                    "mdm_delegate_called": False,
                },
            )
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=active",
                symbol,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "state": "active",
                    "callback_attached": bool(self._tick_subscribers.get(symbol)),
                    "mdm_subscribe_requested": False,
                    "success": False,
                    "reason": "mdm_no_subscribe_callable",
                },
            )

    def flush_pending_live_subscriptions(self) -> int:
        """Activate deferred live subscriptions. Args: none. Returns: flush count. Raises: none."""
        pending_symbols = sorted(self._pending_live_symbols)
        self._defer_live_symbol_subscriptions = False
        if not pending_symbols:
            LOGGER.info(
                "datahub_live_symbol_flush count=0",
                extra={"event": "datahub_live_symbol_flush", "count": 0},
            )
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS flush_count=0",
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "trace_id": self._make_trace_id("__flush__"),
                    "registration_state": "flush_empty",
                    "deferred_mode": False,
                    "callback_attached": False,
                    "mdm_subscribed": False,
                    "pending_count": 0,
                    "reason": "flush_no_pending",
                },
            )
            return 0

        LOGGER.info(
            "datahub_live_symbol_flush count=%d",
            len(pending_symbols),
            extra={"event": "datahub_live_symbol_flush", "count": len(pending_symbols)},
        )
        for symbol in pending_symbols:
            trace_id = self._make_trace_id(symbol)
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=active reason=flush",
                symbol,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "registration_state": "active",
                    "deferred_mode": False,
                    "callback_attached": bool(self._tick_subscribers.get(symbol)),
                    "mdm_subscribed": symbol in self._mdm_subscribed_symbols,
                    "pending_count": len(self._pending_live_symbols),
                    "reason": "flush_activating",
                },
            )
            self._subscribe_symbol(symbol)
        self._pending_live_symbols.clear()
        return len(pending_symbols)

    def subscribe_ticks(self, symbol: str, callback: Optional[TickListener] = None):
        normalized = self._canonical_quote_symbol(symbol)
        if not normalized or normalized.isdigit():
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=rejected_invalid",
                symbol,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": symbol,
                    "trace_id": self._make_trace_id(str(symbol)),
                    "registration_state": "rejected",
                    "deferred_mode": self._defer_live_symbol_subscriptions,
                    "callback_attached": callback is not None,
                    "mdm_subscribed": False,
                    "pending_count": len(self._pending_live_symbols),
                    "reason": "invalid_or_token_symbol",
                },
            )
            return
        trace_id = self._make_trace_id(normalized)
        if callback is not None:
            self._tick_subscribers[normalized].add(callback)
        if self._defer_live_symbol_subscriptions:
            self._pending_live_symbols.add(normalized)
            LOGGER.info(
                "datahub_symbol_registered_deferred symbol=%s",
                normalized,
                extra={
                    "event": "datahub_symbol_registered_deferred",
                    "symbol": normalized,
                },
            )
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=deferred",
                normalized,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": normalized,
                    "trace_id": trace_id,
                    "state": "deferred",
                    "deferred_mode": True,
                    "callback_attached": bool(self._tick_subscribers.get(normalized)),
                    "mdm_subscribe_requested": False,
                    "success": True,
                    "pending_count": len(self._pending_live_symbols),
                    "reason": "deferred_mode_active",
                },
            )
        else:
            LOGGER.info(
                "DATAHUB_SYMBOL_STATUS symbol=%s state=registered",
                normalized,
                extra={
                    "event": "DATAHUB_SYMBOL_STATUS",
                    "symbol": normalized,
                    "trace_id": trace_id,
                    "state": "active",
                    "deferred_mode": False,
                    "callback_attached": bool(self._tick_subscribers.get(normalized)),
                    "mdm_subscribe_requested": True,
                    "success": True,
                    "pending_count": len(self._pending_live_symbols),
                    "reason": "subscribe_immediate",
                },
            )
            self._subscribe_symbol(normalized)
        if callback is not None:
            quote = self.get_quote(normalized, allow_pull=False)
            if quote is not None:
                try:
                    callback(dict(quote))
                except Exception as exc:  # noqa: BLE001
                    self._log_listener_failure(exc)

    def unsubscribe(self, symbol: str, callback: Optional[TickListener] = None):
        self.unsubscribe_ticks(symbol, callback)

    unsubscribe_ticks = unsubscribe

    def unsubscribe_ticks(self, symbol: str, callback: Optional[TickListener] = None):
        normalized = self._canonical_quote_symbol(symbol)
        if callback is not None and callback in self._tick_subscribers.get(normalized, set()):
            self._tick_subscribers[normalized].remove(callback)
        if not self._tick_subscribers.get(normalized):
            self._pending_live_symbols.discard(normalized)
            self._mdm_subscribed_symbols.discard(normalized)
            mdm_unsub = getattr(self._mdm, "unsubscribe", None)
            if callable(mdm_unsub):
                try:
                    mdm_unsub(normalized, self.ingest_tick_sync)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("MDM unsubscribe failed for %s: %s", normalized, exc)

    def subscribe_orders(self, callback) -> None:
        if callback is None:
            return
        self._order_subscribers.append(callback)
        mdm_fn = getattr(self._mdm, "subscribe_orders", None)
        if callable(mdm_fn):
            try:
                mdm_fn(callback)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("subscribe_orders delegate failed: %s", exc)

    def unsubscribe_orders(self, callback) -> None:
        if callback in self._order_subscribers:
            self._order_subscribers.remove(callback)
        mdm_fn = getattr(self._mdm, "unsubscribe_orders", None)
        if callable(mdm_fn):
            try:
                mdm_fn(callback)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("unsubscribe_orders delegate failed: %s", exc)

    def get_latest_price(
        self,
        symbol: str,
        *,
        allow_pull: bool = False,
        allow_rest_fallback: bool | None = None,
    ) -> Optional[float]:
        tick = self.get_quote(symbol, allow_pull=False)
        if tick:
            return self._tick_price(tick)
        mdm_cached = getattr(self._mdm, "get_cached_ltp", None)
        if callable(mdm_cached):
            try:
                cached = mdm_cached(symbol, max_age_seconds=None, require_ws=False)
                if cached is not None:
                    return float(cached)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_cached_ltp delegate failed for %s: %s", symbol, exc)
        if not allow_pull and allow_rest_fallback is not True:
            return None
        mdm_fn = getattr(self._mdm, "get_latest_price", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol, allow_rest_fallback=allow_rest_fallback)
            except TypeError:
                if allow_rest_fallback is True:
                    try:
                        return mdm_fn(symbol)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "get_latest_price legacy delegate failed for %s: %s",
                            symbol,
                            exc,
                        )
                return None
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_latest_price delegate failed for %s: %s", symbol, exc)
        return None

    def get_cached_ltp(
        self,
        symbol: str,
        *,
        max_age_seconds: float | None = None,
        require_ws: bool = False,
    ) -> Optional[float]:
        """Get cached LTP only. Args: symbol/guards. Returns: price. Raises: none."""
        mdm_cached = getattr(self._mdm, "get_cached_ltp", None)
        if callable(mdm_cached):
            return mdm_cached(
                symbol, max_age_seconds=max_age_seconds, require_ws=require_ws
            )
        quote = self.get_quote(symbol, allow_pull=False)
        if not quote:
            return None
        return self._tick_price(quote)

    def pull_quote(self, symbol: str) -> Dict[str, Any]:
        mdm_fn = getattr(self._mdm, "pull_quote", None)
        if callable(mdm_fn):
            try:
                quote = mdm_fn(symbol) or {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("pull_quote delegate failed for %s: %s", symbol, exc)
                return {}
            if quote:
                self.store_quote(symbol, quote, source=str(quote.get("source") or "poll"))
                return self.get_quote(symbol, allow_pull=False) or {}
        return self.get_quote(symbol, allow_pull=False) or {}

    def probe_quote(self, symbol: str) -> Dict[str, Any]:
        mdm_fn = getattr(self._mdm, "probe_quote", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol) or {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("probe_quote delegate failed for %s: %s", symbol, exc)
        return self.get_quote(symbol) or {}

    def get_ohlc_bars(self, symbol: str, *, limit: Optional[int] = None) -> list:
        mdm_fn = getattr(self._mdm, "get_ohlc_bars", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol, limit=limit) if limit is not None else mdm_fn(symbol)
            except TypeError:
                return mdm_fn(symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_ohlc_bars delegate failed for %s: %s", symbol, exc)
        return []

    def ensure_tracking(self, symbol: str, *, seed: bool = True) -> bool:
        mdm_fn = getattr(self._mdm, "ensure_tracking", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn(symbol, seed=seed))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("ensure_tracking delegate failed for %s: %s", symbol, exc)
        return False

    def is_symbol_ready(self, symbol: str) -> bool:
        mdm_fn = getattr(self._mdm, "is_symbol_ready", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn(symbol))
            except Exception:  # noqa: BLE001
                return False
        return self.get_quote(symbol, allow_pull=False) is not None

    def resolve_symbol_token(self, symbol: str) -> Optional[int]:
        normalized = self._canonical_quote_symbol(symbol)
        with self._lock:
            token = self._token_by_symbol.get(normalized)
        if token is not None:
            return token
        mdm_fn = getattr(self._mdm, "resolve_symbol_token", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_option_chain(self, *args, **kwargs):
        mdm_fn = getattr(self._mdm, "get_option_chain", None)
        if callable(mdm_fn):
            return mdm_fn(*args, **kwargs)
        return None

    def process_ticks(self, ticks) -> None:
        mdm_fn = getattr(self._mdm, "process_ticks", None)
        if callable(mdm_fn):
            try:
                mdm_fn(ticks)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("process_ticks delegate failed: %s", exc, exc_info=exc)

    def get_available_balance(self, *, force: bool = False) -> Optional[float]:
        mdm_fn = getattr(self._mdm, "get_available_balance", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(force=force)
            except TypeError:
                try:
                    return mdm_fn()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("get_available_balance fallback failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_available_balance delegate failed: %s", exc)
        snapshot = self.get_account_snapshot(force=False)
        for key in ("available", "available_cash", "net", "cash"):
            value = snapshot.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def get_account_snapshot(self, *, force: bool = False) -> Dict[str, Any]:
        mdm = self._mdm
        cached = getattr(mdm, "_account_snapshot", None)
        if isinstance(cached, dict):
            snapshot = dict(cached)
        else:
            snapshot = {}
        refresher = getattr(mdm, "get_account_snapshot", None)
        if callable(refresher):
            try:
                return dict(refresher(force=force) or snapshot)
            except TypeError:
                try:
                    return dict(refresher() or snapshot)
                except Exception:  # noqa: BLE001
                    return snapshot
            except Exception:  # noqa: BLE001
                return snapshot
        return snapshot

    def is_ready(self, symbol: Optional[Any] = None) -> bool:
        mdm_fn = getattr(self._mdm, "is_ready", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn())
            except Exception:  # noqa: BLE001
                return False
        return bool(self._quotes)

    def is_fresh(self, symbol: str, *, threshold_ms: Optional[float] = None) -> tuple[bool, Dict[str, Any]]:
        sym = self._canonical_quote_symbol(symbol)
        threshold = float(threshold_ms) if threshold_ms is not None else max(self._stale_tick_max_age_ms, 0.0)
        quote = self.get_quote(sym, allow_pull=False)
        if not quote:
            return False, {"reason": "no_tick", "symbol": sym, "threshold_ms": threshold}
        source = str(quote.get("source") or "ws").lower()
        if source == "historical":
            return True, {"source": source, "threshold_ms": threshold, "reason": None}
        now_ms = self._now() * 1000.0
        ts_ms = self._timestamp_ms(quote.get("timestamp"))
        effective_ms = max(0.0, now_ms - ts_ms)
        if (self._monotonic() - self._start_mono) < self._warmup_grace_s:
            return True, {
                "symbol": sym,
                "source": source,
                "effective_ms": effective_ms,
                "threshold_ms": threshold,
                "reason": "warmup_grace",
            }
        fresh = effective_ms <= threshold
        return fresh, {
            "symbol": sym,
            "source": source,
            "effective_ms": effective_ms,
            "threshold_ms": threshold,
            "reason": None if fresh else "stale",
        }

    def get_iv(self, symbol: str) -> Optional[float]:
        normalized = self._canonical_quote_symbol(symbol)
        if normalized in self._iv_cache:
            return self._iv_cache[normalized]
        quote = self.get_quote(normalized)
        if quote is None:
            return None
        self._capture_option_metrics(normalized, quote)
        return self._iv_cache.get(normalized)

    def get_oi(self, symbol: str) -> Optional[float]:
        normalized = self._canonical_quote_symbol(symbol)
        if normalized in self._oi_cache:
            return self._oi_cache[normalized]
        quote = self.get_quote(normalized, allow_pull=False) or {}
        for key in ("oi", "open_interest"):
            value = quote.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def get_greeks(self, symbol: str) -> dict[str, float] | None:
        normalized = self._canonical_quote_symbol(symbol)
        if normalized in self._greeks_cache:
            return dict(self._greeks_cache[normalized])
        quote = self.get_quote(normalized)
        if quote is None:
            return None
        self._capture_option_metrics(normalized, quote)
        greeks = self._greeks_cache.get(normalized)
        return dict(greeks) if greeks else None

    def expected_move(self, days: float) -> Optional[float]:
        spot = None
        cached_fn = getattr(self, "get_cached_ltp", None)
        if callable(cached_fn):
            spot = cached_fn("NSE:NIFTY", max_age_seconds=300.0, require_ws=False)
        if spot is None:
            return None
        iv = next(iter(self._iv_cache.values()), None)
        if iv is None:
            return None
        return float(spot) * float(iv) * sqrt(float(days) / 365.0)

    async def fetch_history(self, symbol: str, interval: str, days: int = 3) -> list[dict]:
        normalized = self._canonical_quote_symbol(symbol)
        key = (normalized, str(interval or "minute").lower(), int(days or 0))
        mdm_fn = getattr(self._mdm, "fetch_history", None)
        if not callable(mdm_fn):
            return [dict(row) for row in self._history_cache.get(key, [])]
        try:
            rows = await mdm_fn(normalized.replace("NSE:", "").replace("NFO:", ""), interval, days)
        except Exception:  # noqa: BLE001
            return [dict(row) for row in self._history_cache.get(key, [])]
        normalized_rows = self._normalize_history_rows(rows)
        if normalized_rows:
            self._history_cache[key] = normalized_rows
        return [dict(row) for row in self._history_cache.get(key, [])]

    def _normalize_history_rows(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)
        current_minute = now_utc.replace(second=0, microsecond=0)
        for row in rows:
            timestamp = row.get("timestamp")
            if isinstance(timestamp, datetime):
                ts = timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            else:
                ts = datetime.fromtimestamp(self._timestamp_ms(timestamp) / 1000.0, tz=timezone.utc).replace(
                    second=0,
                    microsecond=0,
                )
            if ts == current_minute:
                continue
            normalized = dict(row)
            normalized["timestamp"] = ts
            normalized_rows.append(normalized)
        return normalized_rows

    def get_indicator(self, symbol: str, name: str) -> Optional[float]:
        mdm_fn = getattr(self._mdm, "get_indicator", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol, name)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_position_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        mdm_fn = getattr(self._mdm, "get_position_state", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_data(self, token: Any):
        mdm_fn = getattr(self._mdm, "get_data", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(token)
            except Exception:  # noqa: BLE001
                return None, None
        return None, None

    def get_stats(self) -> Dict[str, Any]:
        return self.stats()

    def is_ws_fresh(self, symbol: str, threshold_sec: float = 2.0) -> bool:
        normalized = self._canonical_quote_symbol(symbol)
        with self._lock:
            last = self._last_ws_arrival.get(normalized)
        if not last:
            return False
        return (self._now() - (last / 1000.0)) < threshold_sec

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            last_ws_age = 0.0
            if self._last_global_ws_arrival:
                last_ws_age = self._now() - (self._last_global_ws_arrival / 1000.0)
            return {
                "tokens": len(self._ticks),
                "symbols": len(self._quotes),
                "last_ws_age_sec": last_ws_age,
            }

    def _mdm_call(self, name: str, *args, default: Any = None, **kwargs) -> Any:
        fn = getattr(self._mdm, name, None)
        if not callable(fn):
            return default
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("DataHub MDM delegate %s failed: %s", name, exc)
            return default

    @property
    def ws_connected(self) -> bool:
        return bool(getattr(self._mdm, "ws_connected", False))

    def set_ws_connected(self, connected: bool) -> None:
        self._mdm_call("set_ws_connected", bool(connected))

    def bump_heartbeat(self) -> None:
        self._mdm_call("bump_heartbeat")

    def register_heartbeat_callback(self, cb) -> None:
        self._mdm_call("register_heartbeat_callback", cb)

    def heartbeat_age(self) -> Optional[float]:
        return self._mdm_call("heartbeat_age")

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        return self._mdm_call("get_latest_tick", symbol) or self.get_quote(
            symbol, allow_pull=False
        )

    def get_last_tick(self, symbol: str) -> Optional[Tick]:
        return self._mdm_call("get_last_tick", symbol) or self.get_quote(
            symbol, allow_pull=False
        )

    def get_ltp(self, symbol: str) -> Optional[float]:
        cached = self.get_cached_ltp(symbol, max_age_seconds=None, require_ws=False)
        if cached is not None:
            return float(cached)
        return self.get_latest_price(symbol, allow_pull=False)

    def has_quote(self, symbol: str) -> bool:
        value = self._mdm_call("has_quote", symbol)
        if value is not None:
            return bool(value)
        return self.get_quote(symbol, allow_pull=False) is not None

    def quote_age_ms(self, symbol: str) -> Optional[float]:
        return self._mdm_call("quote_age_ms", symbol)

    def refresh_quote_now(self, symbol: str, *args, **kwargs) -> Any:
        return self._mdm_call("refresh_quote_now", symbol, *args, **kwargs)

    def cached_mid(self, symbol: str) -> Optional[float]:
        return self._mdm_call("cached_mid", symbol)

    def ingest_rest_quote(self, *args, **kwargs) -> Any:
        return self._mdm_call("ingest_rest_quote", *args, **kwargs)

    def register_symbol(self, symbol: str, token: Optional[int] = None) -> Any:
        if token is not None:
            try:
                normalized = self._canonical_quote_symbol(symbol)
                token_int = int(token)
                self._token_by_symbol[normalized] = token_int
                self._symbol_by_token[token_int] = normalized
            except (TypeError, ValueError):
                pass
        return self._mdm_call("register_symbol", symbol, token)

    def untrack(self, symbol: str) -> Any:
        return self._mdm_call("untrack", symbol)

    def is_tracked(self, symbol: str) -> bool:
        return bool(self._mdm_call("is_tracked", symbol, default=False))

    def list_tracked(self) -> list:
        return list(self._mdm_call("list_tracked", default=[]) or [])

    def tracked_snapshot(self) -> Dict[str, Any]:
        return dict(self._mdm_call("tracked_snapshot", default={}) or {})

    def wait_for_symbol(self, *args, **kwargs) -> Any:
        return self._mdm_call("wait_for_symbol", *args, **kwargs)

    def update_hydration_status(self, *args, **kwargs) -> Any:
        return self._mdm_call("update_hydration_status", *args, **kwargs)

    def get_hydration_status(self, *args, **kwargs) -> Any:
        return self._mdm_call("get_hydration_status", *args, **kwargs)

    def get_ohlc(self, *args, **kwargs) -> Any:
        return self._mdm_call("get_ohlc", *args, **kwargs)

    def get_orderbook(self, *args, **kwargs) -> Any:
        return self._mdm_call("get_orderbook", *args, **kwargs)

    def refresh_margin_snapshot(self, *args, **kwargs) -> Any:
        return self._mdm_call("refresh_margin_snapshot", *args, **kwargs)

    def get_margin_snapshot(self, *args, **kwargs) -> Any:
        return self._mdm_call("get_margin_snapshot", *args, **kwargs)

    def is_live(self) -> bool:
        return bool(self._mdm_call("is_live", default=False))

    def is_data_stale(self, *args, **kwargs) -> bool:
        return bool(self._mdm_call("is_data_stale", *args, default=False, **kwargs))

    def transport_status(self) -> Dict[str, Any]:
        return dict(self._mdm_call("transport_status", default={}) or {})

    def mdm_status(self) -> Dict[str, Any]:
        return dict(self._mdm_call("mdm_status", default={}) or {})


__all__ = ["DataHub", "Tick", "TickBus", "TickListener", "OrderListener"]
