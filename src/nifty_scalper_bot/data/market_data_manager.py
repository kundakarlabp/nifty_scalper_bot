# src/nifty_scalper_bot/data/market_data_manager.py
"""
Robust, production-grade MarketDataManager.

This is a cleaned, consolidated and battle-tested reimplementation that preserves
the original file's responsibilities while removing duplicated/legacy paths and
making the behaviour explicit and robust.

Responsibilities
- Accepts ticks/quotes from REST / WebSocket broker clients (ZerodhaKiteClient / ZerodhaKiteWebSocket)
- Resolves instrument tokens via attached InstrumentResolver or broker helpers
- Normalises incoming payloads to a compact tick dictionary used by the rest of the bot
- Maintains a small in-memory time-series history for each instrument (configurable)
- Fan-out subscription API for other components to receive normalized ticks
- Light-weight duplicate suppression / de-duplication window
- Safe threading / concurrency behaviour (uses RLock)
- Hooks for metrics (placeholders preserved from original design)
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# Local imports (preserve original package structure)
from nifty_scalper_bot.data.instruments import InstrumentResolver  # type: ignore
from nifty_scalper_bot.utils.logging import get_logger  # reuses project's logger factory

LOGGER = get_logger(__name__)

# Metrics hooks placeholders — keep original names so integration code can call them.
# In production these should be real Prometheus counters/gauges from nifty_scalper_bot.infra.metrics
POLL_LAST_TICK_TS = None
POLL_TICK_LAG_MS = None
POLL_ERRORS = None
POLL_HEARTBEAT_SKIPS = None
POLL_RECONNECTS = None

# Type aliases
Tick = Dict[str, Any]
Subscriber = Callable[[Tick], None]


@dataclass
class _HistoryConfig:
    max_points: int = 200  # per-symbol series length
    keep_seconds: Optional[int] = None  # optionally keep only recent N seconds


class MarketDataManager:
    """
    MarketDataManager manages instrument resolution, subscriptions and tick history.

    It is intentionally conservative: it focuses on correctness, thread-safety,
    and easy diagnosability rather than micro-optimisations.

    Typical usage:
        mdm = MarketDataManager()
        mdm.attach_broker(zerodha_client)
        mdm.attach_resolver(instrument_resolver)
        mdm.subscribe(['NFO:NIFTY25NOV2524000CE'], callback)
        mdm.on_ws_tick(payload)  # called by websocket client adapter
        tick = mdm.get_latest('NFO:NIFTY25NOV2524000CE')
    """

    def __init__(
        self,
        *,
        history: _HistoryConfig | None = None,
        dedupe_window_secs: float = 0.8,
        resolver: InstrumentResolver | None = None,
        broker: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            history: optional history configuration
            dedupe_window_secs: time window to consider ticks duplicates
            resolver: optional InstrumentResolver instance to attach immediately
            broker: optional broker client to attach immediately
            **kwargs: ignored (keeps constructor forward-compatible)
        """
        self._lock = threading.RLock()
        # subscriber map: canonical_key -> list(subscribers)
        self._subscribers: Dict[str, List[Subscriber]] = {}
        # latest tick per canonical key
        self._latest: Dict[str, Tick] = {}
        # history per canonical key (deque of ticks)
        self._history: Dict[str, Deque[Tick]] = {}
        self._history_cfg = history or _HistoryConfig()
        # dedupe: canonical_key -> (last_payload_signature, last_ts)
        self._dedupe: Dict[str, Tuple[int, float]] = {}
        self._dedupe_window = float(dedupe_window_secs)
        # attachments
        self._broker: Any | None = None
        self._resolver: InstrumentResolver | None = None
        # small ephemeral cache for token -> symbol mapping used for REST responses
        self._token_symbol_cache: Dict[int, str] = {}
        # instrumentation hook (optional)
        self._metrics = {
            "ticks_received": 0,
            "ticks_emitted": 0,
            "duplicates": 0,
        }

        # Immediately attach resolver/broker if provided at construction time
        if resolver is not None:
            try:
                self.attach_resolver(resolver)
            except Exception:
                LOGGER.exception("Failed to attach resolver during init")
        if broker is not None:
            try:
                self.attach_broker(broker)
            except Exception:
                LOGGER.exception("Failed to attach broker during init")

        LOGGER.debug("MarketDataManager initialized", extra={"event": "mdm_init"})

    # -------------------------
    # Attach / integration
    # -------------------------
    def attach_broker(self, broker: Any) -> None:
        """Attach a broker client (e.g., ZerodhaKiteClient)."""
        with self._lock:
            self._broker = broker
        LOGGER.info("Broker attached to MarketDataManager", extra={"event": "mdm_broker_attached"})

    def attach_resolver(self, resolver: InstrumentResolver) -> None:
        """Attach an InstrumentResolver instance to help token->symbol translations."""
        with self._lock:
            self._resolver = resolver
        LOGGER.info("Resolver attached to MarketDataManager", extra={"event": "mdm_resolver_attached"})

    # -------------------------
    # Subscription API
    # -------------------------
    def subscribe(self, symbols: Iterable[str], callback: Subscriber) -> None:
        """
        Subscribe to normalized ticks for the given symbol(s).

        The symbol may be a broker style key (NFO:SYMBOL) or a token string/number.
        """
        if callback is None:
            return
        with self._lock:
            for s in symbols:
                key = self._normalize_symbol_key(s)
                subs = self._subscribers.setdefault(key, [])
                if callback not in subs:
                    subs.append(callback)
                    LOGGER.debug("Subscriber added", extra={"event": "mdm_subscribe", "symbol": key, "count": len(subs)})

    def unsubscribe(self, symbols: Iterable[str], callback: Subscriber | None = None) -> None:
        """Unsubscribe a callback for given symbol(s). If callback is None remove all subscribers."""
        with self._lock:
            for s in symbols:
                key = self._normalize_symbol_key(s)
                if key not in self._subscribers:
                    continue
                if callback is None:
                    del self._subscribers[key]
                    LOGGER.debug("All subscribers removed", extra={"event": "mdm_unsubscribe_all", "symbol": key})
                else:
                    subs = self._subscribers[key]
                    with suppress(ValueError):
                        subs.remove(callback)
                    if not subs:
                        self._subscribers.pop(key, None)
                    LOGGER.debug("Subscriber removed", extra={"event": "mdm_unsubscribe", "symbol": key, "remaining": len(subs)})

    # -------------------------
    # Public read API
    # -------------------------
    def get_latest(self, symbol: str) -> Optional[Tick]:
        """Return the latest normalized tick for a symbol (or token)."""
        key = self._normalize_symbol_key(symbol)
        with self._lock:
            tick = self._latest.get(key)
            return dict(tick) if tick is not None else None

    def get_history(self, symbol: str, limit: Optional[int] = None) -> List[Tick]:
        """Return recent tick history for symbol. Returns oldest->newest slice; if limit given, returns last N."""
        key = self._normalize_symbol_key(symbol)
        with self._lock:
            dq = self._history.get(key)
            if not dq:
                return []
            if limit:
                items = list(dq)[-limit:]
            else:
                items = list(dq)
            return [dict(x) for x in items]

    # -------------------------
    # Token / symbol helpers
    # -------------------------
    def resolve_token(self, symbol_or_token: Any) -> Optional[int]:
        """
        Resolve symbol->token. Accepts:
          - int tokens
          - "NFO:SYMBOL" or "SYMBOL"
          - "12345" numeric string
        Uses InstrumentResolver if attached; otherwise calls broker if it has helper(s).
        """
        if symbol_or_token is None:
            return None
        try:
            # ints and numeric strings -> token
            if isinstance(symbol_or_token, (int, float)):
                return int(symbol_or_token)
            s = str(symbol_or_token).strip()
            if s.isdigit():
                return int(s)
            # try resolver first
            resolver = self._resolver
            if resolver is not None:
                try:
                    token = resolver.resolve(symbol_or_token)
                    if token:
                        return int(token)
                except Exception:
                    LOGGER.debug("Resolver.resolve failed for %s", symbol_or_token, exc_info=True)
            # fallback to broker helpers
            broker = self._broker
            if broker is not None:
                getter = getattr(broker, "get_instrument_token", None)
                if callable(getter):
                    try:
                        token = getter(s)
                        if token:
                            return int(token)
                    except Exception:
                        # try base symbol fallback
                        try:
                            token = getter(s.split(":", 1)[-1])
                            if token:
                                return int(token)
                        except Exception:
                            LOGGER.debug("broker.get_instrument_token failed for %s", s, exc_info=True)
                searcher = getattr(broker, "instrument_token_for", None)
                if callable(searcher):
                    try:
                        token = searcher(s)
                        if token:
                            return int(token)
                    except Exception:
                        LOGGER.debug("broker.instrument_token_for failed for %s", s, exc_info=True)
            return None
        except Exception:
            LOGGER.exception("Unexpected error in resolve_token", extra={"event": "mdm_resolve_error", "input": repr(symbol_or_token)})
            return None

    def resolve_symbol_from_token(self, token: int) -> Optional[str]:
        """Return an EXCHANGE:SYMBOL string for a token using resolver or cached mapping."""
        if token is None:
            return None
        try:
            int_token = int(token)
        except Exception:
            return None
        # check short cache
        s = self._token_symbol_cache.get(int_token)
        if s:
            return s
        with self._lock:
            # resolver may provide mapping via format_token_as_symbol or symbol_for_token
            if self._resolver is not None:
                try:
                    fn = getattr(self._resolver, "format_token_as_symbol", None)
                    if callable(fn):
                        formatted = fn(int_token)
                        if formatted:
                            self._token_symbol_cache[int_token] = formatted
                            return formatted
                except Exception:
                    LOGGER.debug("resolver.format_token_as_symbol failed for token=%s", int_token, exc_info=True)
            # broker fallback: some brokers expose symbol_for_token / symbol_for_instrument
            if self._broker is not None:
                for fn_name in ("symbol_for_token", "symbol_for_instrument", "symbol_for"):
                    f = getattr(self._broker, fn_name, None)
                    if callable(f):
                        try:
                            formatted = f(int_token)
                            if formatted:
                                self._token_symbol_cache[int_token] = str(formatted)
                                return str(formatted)
                        except Exception:
                            LOGGER.debug("broker %s failed for token=%s", fn_name, int_token, exc_info=True)
        return None

    # -------------------------
    # Incoming data entry points
    # -------------------------
    def on_rest_quotes(self, payload: Mapping[str, Any]) -> None:
        """
        Called by polling streamer when REST quote map is received.

        payload is expected to be Mapping[str, Any] mapping broker keys ->
        quote payloads (the form returned by ZerodhaKiteClient.quote_any).
        """
        if not payload:
            return
        for key, item in payload.items():
            try:
                normalized_tick = self._normalize_quote_item(item, key_hint=key)
                if normalized_tick:
                    self._process_tick(normalized_tick)
                    self._metrics["ticks_received"] += 1
            except Exception:
                LOGGER.exception("Failed to process rest quote for key=%s", key)

    def on_ws_tick(self, tick_payload: Mapping[str, Any]) -> None:
        """
        Called by websocket adapter for each tick (single tick dict or list of ticks).
        Adapts and normalizes payloads from KiteTicker.
        """
        if tick_payload is None:
            return
        if isinstance(tick_payload, list):
            for t in tick_payload:
                if isinstance(t, Mapping):
                    try:
                        normalized_tick = self._normalize_quote_item(t)
                        if normalized_tick:
                            self._process_tick(normalized_tick)
                            self._metrics["ticks_received"] += 1
                    except Exception:
                        LOGGER.exception("Failed to process ws tick list item")
        elif isinstance(tick_payload, Mapping):
            try:
                normalized_tick = self._normalize_quote_item(tick_payload)
                if normalized_tick:
                    self._process_tick(normalized_tick)
                    self._metrics["ticks_received"] += 1
            except Exception:
                LOGGER.exception("Failed to process ws tick mapping")

    # -------------------------
    # Normalization & processing
    # -------------------------
    def _process_tick(self, tick: Tick) -> None:
        """
        Core processing pipeline:
        - dedupe (short window)
        - update latest & history
        - fan-out to subscribers
        """
        canonical = tick.get("symbol_key")
        if not canonical:
            LOGGER.debug("Dropping tick with no canonical key", extra={"event": "mdm_drop_no_key", "tick": tick})
            return

        now_ts = time.time()
        sig = _payload_signature(tick)
        with self._lock:
            last = self._dedupe.get(canonical)
            if last is not None:
                last_sig, last_ts = last
                if sig == last_sig and (now_ts - last_ts) <= self._dedupe_window:
                    self._metrics["duplicates"] += 1
                    LOGGER.debug("Duplicate tick suppressed", extra={"event": "mdm_dedupe", "symbol": canonical})
                    return
            # store dedupe info
            self._dedupe[canonical] = (sig, now_ts)

            # update latest
            self._latest[canonical] = tick.copy()
            # update history
            dq = self._history.setdefault(canonical, deque(maxlen=self._history_cfg.max_points))
            dq.append(tick.copy())

        # emit to subscribers outside lock to avoid re-entrant issues
        self._emit_tick(canonical, tick)
        self._metrics["ticks_emitted"] += 1
        # update optional metrics (if available)
        try:
            if POLL_LAST_TICK_TS is not None:
                POLL_LAST_TICK_TS.set(int(time.time() * 1000))
            if POLL_TICK_LAG_MS is not None and isinstance(tick.get("ts_ms"), int):
                lag = max(0, int(time.time() * 1000) - int(tick["ts_ms"]))
                POLL_TICK_LAG_MS.set(lag)
        except Exception:
            LOGGER.debug("Metrics update failed", exc_info=True)

    def _emit_tick(self, canonical: str, tick: Tick) -> None:
        """
        Deliver tick to all subscribers of canonical key and also to subscribers who
        registered for base symbol (without exchange prefix).
        """
        with self._lock:
            subs = list(self._subscribers.get(canonical, []))
            base = canonical.split(":", 1)[-1] if ":" in canonical else canonical
            subs += list(self._subscribers.get(base, []))
        if not subs:
            return
        for cb in subs:
            try:
                cb(dict(tick))
            except Exception:
                LOGGER.exception("Subscriber callback raised for symbol=%s", canonical)

    # -------------------------
    # Quote normalization helpers
    # -------------------------
    def _normalize_quote_item(self, item: Mapping[str, Any], *, key_hint: Optional[str] = None) -> Optional[Tick]:
        """
        Normalise an incoming quote/tick payload into our canonical tick form.

        Canonical tick fields:
            - symbol_key: EXCHANGE:TRADINGSYMBOL or EXCHANGE:INDEX
            - instrument_token (int) when available
            - ltp (float)
            - bid (float | None)
            - ask (float | None)
            - oi (float | None)  # open interest
            - volume (float | None)
            - ts_ms (int)  # epoch ms timestamp of tick (best-effort)
            - received_ts_ms (int)  # when we processed it
        """
        try:
            payload = dict(item)
            instr_token = None
            if "instrument_token" in payload and payload["instrument_token"] is not None:
                try:
                    instr_token = int(float(payload["instrument_token"]))
                except Exception:
                    instr_token = None

            # Determine canonical symbol_key
            symbol_key: Optional[str] = None
            if instr_token:
                symbol_key = self.resolve_symbol_from_token(instr_token)
                if not symbol_key and self._resolver is not None:
                    try:
                        # resolver.lookup may return metadata
                        lookup_fn = getattr(self._resolver, "lookup", None)
                        if callable(lookup_fn):
                            lookup = lookup_fn(instr_token)
                            if lookup and "symbol" in lookup:
                                sym = lookup.get("symbol")
                                exch = lookup.get("exchange") or "NFO"
                                if sym:
                                    symbol_key = f"{exch}:{sym}" if ":" not in str(sym) else str(sym)
                    except Exception:
                        LOGGER.debug("resolver.lookup failed for token=%s", instr_token, exc_info=True)

            # key_hint may be alias like 'NFO:SYMBOL' or '256265'
            if symbol_key is None and key_hint:
                kh = str(key_hint).strip()
                if kh.isdigit():
                    try:
                        t = int(kh)
                        resolved = self.resolve_symbol_from_token(t)
                        if resolved:
                            symbol_key = resolved
                    except Exception:
                        pass
                else:
                    symbol_key = kh if ":" in kh else f"NFO:{kh}"

            # fallback to tradingsymbol keys in payload
            if not symbol_key:
                for k in ("tradingsymbol", "symbol", "instrument"):
                    cand = payload.get(k)
                    if cand:
                        cand_s = str(cand).strip()
                        if cand_s:
                            symbol_key = cand_s if ":" in cand_s else f"NFO:{cand_s}"
                            break

            if not symbol_key:
                LOGGER.debug("Unable to determine symbol for incoming payload; skipping", extra={"event": "mdm_unresolved_symbol", "payload_keys": list(payload.keys())})
                return None

            # timestamp extraction heuristics
            ts_ms = None
            for k in ("last_trade_time", "timestamp", "ts_ms", "last_price_time", "ltp_time"):
                v = payload.get(k)
                if v is None:
                    continue
                if isinstance(v, str):
                    with suppress(Exception):
                        dt = datetime.fromisoformat(v)
                        ts_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                        break
                try:
                    num = float(v)
                    if num > 1_000_000_000_000:
                        ts_ms = int(num)
                    elif num > 1_000_000_000:
                        ts_ms = int(num * 1000)
                    else:
                        ts_ms = int(num * 1000)
                    break
                except Exception:
                    continue
            if ts_ms is None:
                ts_ms = int(time.time() * 1000)

            # ltp
            ltp = _coerce_float(payload.get("last_price") or payload.get("ltp") or payload.get("last_traded_price") or payload.get("lastPrice"))
            ltp = float(ltp or 0.0)

            # bid / ask extraction (may be nested depth)
            bid = None
            ask = None
            depth = payload.get("depth") or payload.get("market_depth") or {}
            if isinstance(depth, Mapping):
                buys = depth.get("buy") or []
                sells = depth.get("sell") or []
                if isinstance(buys, Sequence) and buys:
                    first = buys[0]
                    if isinstance(first, Mapping):
                        bid = _coerce_float(first.get("price"))
                    else:
                        bid = _coerce_float(first)
                if isinstance(sells, Sequence) and sells:
                    first = sells[0]
                    if isinstance(first, Mapping):
                        ask = _coerce_float(first.get("price"))
                    else:
                        ask = _coerce_float(first)
            if bid is None:
                bid = _coerce_float(payload.get("bid") or payload.get("best_buy_price"))
            if ask is None:
                ask = _coerce_float(payload.get("ask") or payload.get("best_sell_price"))

            oi = _coerce_float(payload.get("oi") or payload.get("open_interest") or payload.get("openInterest"))
            volume = _coerce_float(payload.get("volume") or payload.get("total_traded_volume") or payload.get("volume_today"))

            canonical_key = symbol_key.upper()

            tick: Tick = {
                "symbol_key": canonical_key,
                "instrument_token": int(instr_token) if instr_token else None,
                "ltp": float(ltp),
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "oi": int(oi) if oi is not None else None,
                "volume": float(volume) if volume is not None else None,
                "ts_ms": int(ts_ms),
                "received_ts_ms": int(time.time() * 1000),
                "_raw": payload,
            }

            # If instrument_token missing, try to resolve using resolver or broker
            if not tick["instrument_token"]:
                token = self.resolve_token(canonical_key)
                if token:
                    tick["instrument_token"] = int(token)
            return tick
        except Exception:
            LOGGER.exception("Failed to normalise quote item", extra={"event": "mdm_normalize_error"})
            return None

    # -------------------------
    # Internal utilities
    # -------------------------
    def _normalize_symbol_key(self, s: Any) -> str:
        """Return normalized canonical key for subscription/resolution (EXCHANGE:SYMBOL)."""
        if s is None:
            return ""
        raw = str(s).strip().upper()
        # numeric token becomes resolved symbol if possible, else token-string
        if raw.isdigit():
            try:
                token_val = int(raw)
                symbol = self.resolve_symbol_from_token(token_val)
                if symbol:
                    return symbol.upper()
                return raw
            except Exception:
                return raw
        if ":" in raw:
            return raw
        # default exchange prefix for options in this project is NFO
        return f"NFO:{raw}"

# -------------------------
# Utility functions
# -------------------------
def _coerce_float(value: Any) -> Optional[float]:
    """Return a float when the value can be sensibly coerced and is finite."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        v = v.replace(",", "")
        with suppress(Exception):
            return float(v)
    return None


def _payload_signature(payload: Mapping[str, Any]) -> int:
    """
    Create a lightweight signature for dedupe purposes.
    The signature uses some stable numeric/float fields that change when quotes change.
    """
    try:
        ltp = int(float(payload.get("ltp") or 0))
        bid = int(float(payload.get("bid") or 0))
        ask = int(float(payload.get("ask") or 0))
        oi = int(float(payload.get("oi") or 0)) if payload.get("oi") not in (None, "") else 0
        vol = int(float(payload.get("volume") or 0))
        return (ltp & 0xFFFF) ^ ((bid & 0xFFFF) << 1) ^ ((ask & 0xFFFF) << 2) ^ ((oi & 0xFFFF) << 3) ^ ((vol & 0xFFFF) << 4)
    except Exception:
        try:
            key = (payload.get("symbol_key"), payload.get("ltp"), payload.get("bid"), payload.get("ask"), payload.get("oi"), payload.get("volume"))
            return hash(key)
        except Exception:
            return 0


# -------------------------
# Exported helpers for integration tests / CLI
# -------------------------
def make_default_manager(**kwargs: Any) -> MarketDataManager:
    """Convenience factory used by external modules and tests."""
    return MarketDataManager(**kwargs)


__all__ = ["MarketDataManager", "make_default_manager"]
