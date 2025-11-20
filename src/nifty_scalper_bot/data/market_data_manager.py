# src/nifty_scalper_bot/data/market_data_manager.py
"""
MarketDataManager
-----------------
Robust production-ready market data manager that supports:

- broker REST quote lookups (batch & single)
- websocket subscription integration (if websocket_manager provided)
- polling fallback (background thread polling)
- token/symbol resolution via InstrumentResolver (if attached)
- light caching and thread-safety

Constructor signature intentionally compatible with the rest of the codebase:
    MarketDataManager(broker_client, websocket_manager=None, resolver=None, *, poll_interval=0.7)

This file is designed to be a reliable single-file improvement that is:
- defensive (no uncaught exceptions from background threads)
- thread-safe (RLock around caches)
- straightforward to extend
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

LOGGER = logging.getLogger("nifty_scalper_bot.data.market_data_manager")


class MarketDataManager:
    """
    Manage market-data access via broker REST and optionally a websocket manager.

    Parameters
    ----------
    broker_client : Any
        Broker client implementing quote_any, get_quote, get_quote_bulk, list_instruments/get_instruments/get_instrument_token, etc.
    websocket_manager : Any | None
        Optional websocket manager object capable of subscribe/unsubscribe/set_mode/attach callbacks.
        If provided, MarketDataManager will try to use it for streaming and fall back to polling where needed.
    resolver : InstrumentResolver | None
        Optional instrument resolver used to map tokens <-> symbols and produce tradingsymbols for orders.
    poll_interval : float
        When polling mode is active, interval between poll batches (seconds).
    """

    def __init__(
        self,
        broker_client: Any,
        websocket_manager: Any | None = None,
        resolver: Any | None = None,
        *,
        poll_interval: float = 0.7,
        default_exchange: str = "NFO",
    ) -> None:
        # external dependencies
        self._broker = broker_client
        self._ws = websocket_manager
        self._resolver = resolver

        # configuration
        self._poll_interval = float(poll_interval)
        self._default_exchange = (default_exchange or "NFO").upper()

        # runtime caches and guards
        self._lock = threading.RLock()
        # token -> symbol like "NFO:NIFTY25NOV2524000CE" or "NFO:NIFTY 50"
        self._token_to_symbol: Dict[int, str] = {}
        # symbol -> token
        self._symbol_to_token: Dict[str, int] = {}
        # alias map returned by quote_any to map bare/truncated keys to canonical
        self._last_quote_snapshot: Dict[str, Mapping[str, Any]] = {}

        # Polling background thread controls
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_polling = False
        self._poll_targets: List[int] = []  # instrument tokens polled in batch
        self._poll_lock = threading.RLock()

        # streaming mode: "auto"/"websocket"/"polling"
        self._streaming_mode = "auto"

        # if websocket manager provided, attempt to wire callbacks
        if self._ws is not None:
            self._attach_ws_handlers(self._ws)

        LOGGER.info(
            "MarketDataManager initialized",
            extra={
                "event": "market_data_manager_init",
                "has_websocket": bool(self._ws),
                "poll_interval": self._poll_interval,
                "streaming_mode": self._streaming_mode,
            },
        )

    # -----------------------
    # Public API (expected by app)
    # -----------------------
    def attach_resolver(self, resolver: Any) -> None:
        """Attach an InstrumentResolver instance (or compatible object)."""
        with self._lock:
            self._resolver = resolver
            # Try warm caches if resolver already has some mapping methods
            try:
                # If resolver exposes reverse mapping we can warm our caches
                if hasattr(resolver, "instrument_catalog") and callable(
                    getattr(resolver, "instrument_catalog")
                ):
                    catalog = resolver.instrument_catalog() or {}
                    for k, v in getattr(catalog, "items", lambda: [])():
                        # skip if not mapping
                        if isinstance(v, Mapping):
                            token = self._extract_token_from_mapping(v)
                            if token:
                                self._register_token_symbol(int(token), k)
            except Exception:
                LOGGER.debug("attach_resolver warm attempt failed", exc_info=True)

    def set_streaming_mode(self, mode: str) -> None:
        """
        Set streaming mode. Accepts "auto", "websocket", "polling".
        'auto' tries websocket then polling.
        """
        mode_clean = (mode or "auto").strip().lower()
        if mode_clean not in ("auto", "websocket", "polling"):
            raise ValueError("streaming mode must be one of 'auto', 'websocket', 'polling'")
        self._streaming_mode = mode_clean
        LOGGER.info("market_data_streaming_mode_set", extra={"event": "market_data_streaming_mode_set", "mode": mode_clean})

    def quote_any(self, items: Iterable[object]) -> Optional[Mapping[str, Any]]:
        """
        Query broker for mixed identifiers (tokens or EXCHANGE:SYMBOL).
        Delegates to broker.quote_any when available.
        Returns a mapping keyed by requested identifier (or None).
        """
        items_list = list(items or [])
        if not items_list:
            return None

        # Prefer broker.quote_any if available
        qfn = getattr(self._broker, "quote_any", None)
        if callable(qfn):
            try:
                return qfn(items_list)
            except Exception:
                LOGGER.exception("broker.quote_any failed", extra={"event": "market_data_quote_any_error"})
                # fallback below

        # Fallback: try individual get_quote / get_quote_by_token
        out: Dict[str, Any] = {}
        for itm in items_list:
            try:
                if isinstance(itm, (int, float)) or (isinstance(itm, str) and str(itm).strip().isdigit()):
                    token = int(itm)
                    payload = self.get_quote_by_token(token)
                    if payload is not None:
                        out[str(token)] = payload
                else:
                    # ensure EXCHANGE:SYMBOL format for broker.get_quote
                    txt = str(itm).strip()
                    if ":" not in txt:
                        txt = f"{self._default_exchange}:{txt}"
                    payload = self.get_quote(txt)
                    if payload is not None:
                        out[txt] = payload
            except Exception:
                LOGGER.exception("quote_any fallback per-item failed", extra={"event": "market_data_quote_any_item_error", "item": itm})
        return out or None

    def get_quote(self, symbol: str) -> Mapping[str, Any] | None:
        """
        Get normalized quote for EXCHANGE:SYMBOL or SYMBOL.
        Tries broker.get_quote, then quote_any fallback.
        """
        if not symbol:
            return None
        try:
            qfn = getattr(self._broker, "get_quote", None)
            kite_sym = symbol if ":" in symbol else f"{self._default_exchange}:{symbol}"
            if callable(qfn):
                try:
                    return qfn(kite_sym)
                except Exception:
                    LOGGER.debug("broker.get_quote failed for %s; falling back to quote_any", kite_sym, exc_info=True)
            # fallback to quote_any
            res = self.quote_any([kite_sym])
            if res:
                return res.get(kite_sym) or res.get(kite_sym.split(":", 1)[-1])
        except Exception:
            LOGGER.exception("get_quote failed", extra={"event": "market_data_get_quote_error", "symbol": symbol})
        return None

    def get_quote_by_token(self, token: int) -> Optional[Mapping[str, Any]]:
        """Return broker quote payload for a token, with instrument_token populated if possible."""
        if token is None:
            return None
        try:
            # prefer broker.get_quote_by_token or get_quote_bulk if present
            get_token_fn = getattr(self._broker, "get_quote_by_token", None)
            if callable(get_token_fn):
                payload = get_token_fn(int(token))
                if isinstance(payload, Mapping):
                    payload = dict(payload)
                    payload.setdefault("instrument_token", int(token))
                    return payload
            # fallback to get_quote_bulk via symbol translation
            symbol_list, sym_map = self._tokens_to_symbols([int(token)])
            if symbol_list:
                response = self.quote_any(symbol_list)
                if response:
                    # attempt by symbol, else by token key
                    payload = response.get(symbol_list[0]) or response.get(str(int(token)))
                    if isinstance(payload, Mapping):
                        payload = dict(payload)
                        payload.setdefault("instrument_token", int(token))
                        return payload
        except Exception:
            LOGGER.exception("get_quote_by_token failed", extra={"event": "market_data_get_quote_by_token_error", "token": token})
        return None

    def get_quote_bulk(self, tokens: Iterable[int]) -> Dict[int, Mapping[str, Any]]:
        """
        Return mapping token -> broker quote payloads. Uses broker.get_quote_bulk or quote_any.
        """
        tokens_list = [int(t) for t in (tokens or []) if t is not None]
        if not tokens_list:
            return {}
        try:
            # Try broker.get_quote_bulk (some clients implement)
            get_bulk = getattr(self._broker, "get_quote_bulk", None)
            if callable(get_bulk):
                try:
                    raw = get_bulk(tokens_list)
                    # Expecting mapping token -> payload
                    out = {}
                    for k, v in (raw or {}).items():
                        try:
                            tok = int(k)
                        except Exception:
                            tok = int(v.get("instrument_token") or 0) if isinstance(v, Mapping) else 0
                        if tok:
                            out[int(tok)] = v
                    return out
                except Exception:
                    LOGGER.debug("broker.get_quote_bulk failed; falling back to quote_any", exc_info=True)
            # Fallback: convert tokens -> symbols and call quote_any
            symbols, symbol_map = self._tokens_to_symbols(tokens_list)
            if not symbols:
                return {}
            response = self.quote_any(symbols)
            out: Dict[int, Mapping[str, Any]] = {}
            if response:
                for sym, payload in response.items():
                    # try to determine the token for payload
                    if isinstance(payload, Mapping):
                        token_val = payload.get("instrument_token") or symbol_map.get(sym) or 0
                        try:
                            t = int(token_val)
                        except Exception:
                            t = 0
                        if t:
                            out[t] = payload
                # also ensure tokens requested but missing are represented (optional)
            return out
        except Exception:
            LOGGER.exception("get_quote_bulk failed", extra={"event": "market_data_get_quote_bulk_error", "tokens": tokens_list})
            return {}

    def get_ltp_bulk(self, tokens: Iterable[int]) -> Dict[int, float]:
        """
        Return mapping token -> last traded price (LTP) using best available endpoint.
        """
        out: Dict[int, float] = {}
        try:
            # If broker exposes get_ltp or get_ltp_bulk, prefer it
            get_ltp_bulk_fn = getattr(self._broker, "get_ltp_bulk", None)
            if callable(get_ltp_bulk_fn):
                try:
                    return get_ltp_bulk_fn(list(tokens))
                except Exception:
                    LOGGER.debug("broker.get_ltp_bulk failed, falling back", exc_info=True)
            # fallback to get_quote_bulk -> extract last_price
            quote_map = self.get_quote_bulk(tokens)
            for tok, payload in quote_map.items():
                try:
                    lp = 0.0
                    if isinstance(payload, Mapping):
                        lp = float(payload.get("last_price") or payload.get("ltp") or 0.0)
                    if lp > 0:
                        out[int(tok)] = float(lp)
                except Exception:
                    continue
        except Exception:
            LOGGER.exception("get_ltp_bulk failed", extra={"event": "market_data_get_ltp_bulk_error"})
        return out

    def subscribe(self, instrument_tokens: Iterable[int]) -> None:
        """
        Subscribe to instrument tokens via websocket manager if present; otherwise register them for polling.
        """
        tokens = [int(t) for t in (instrument_tokens or []) if t is not None]
        if not tokens:
            return
        with self._lock:
            if self._ws is not None and hasattr(self._ws, "subscribe"):
                try:
                    self._ws.subscribe(tokens)
                    LOGGER.debug("ws_subscribe", extra={"event": "market_data_ws_subscribe", "tokens": tokens})
                    return
                except Exception:
                    LOGGER.exception("Websocket subscribe failed, falling back to polling", extra={"event": "market_data_ws_subscribe_error", "tokens": tokens})
            # fallback to polling: add to poll targets and ensure poll thread running
            with self._poll_lock:
                for t in tokens:
                    if t not in self._poll_targets:
                        self._poll_targets.append(t)
                if not self._is_polling:
                    self._start_polling_thread()

    def unsubscribe(self, instrument_tokens: Iterable[int]) -> None:
        """Unsubscribe tokens from websocket or remove from polling targets."""
        tokens = [int(t) for t in (instrument_tokens or []) if t is not None]
        if not tokens:
            return
        with self._lock:
            if self._ws is not None and hasattr(self._ws, "unsubscribe"):
                try:
                    self._ws.unsubscribe(tokens)
                    LOGGER.debug("ws_unsubscribe", extra={"event": "market_data_ws_unsubscribe", "tokens": tokens})
                    return
                except Exception:
                    LOGGER.exception("Websocket unsubscribe failed, falling back to polling removal", extra={"event": "market_data_ws_unsubscribe_error", "tokens": tokens})
            with self._poll_lock:
                for t in tokens:
                    with suppress(ValueError):
                        self._poll_targets.remove(t)
                LOGGER.debug("poll_unsubscribe", extra={"event": "market_data_poll_unsubscribe", "remaining": len(self._poll_targets)})

    # -----------------------
    # Utilities for resolver/broker mapping
    # -----------------------
    def _tokens_to_symbols(self, tokens: Iterable[int]) -> Tuple[List[str], Dict[str, int]]:
        """
        Convert int tokens -> list of EXCHANGE:SYMBOL strings and a symbol_map.
        Uses the attached resolver if present, otherwise uses cached mapping and broker fallback.
        """
        out_symbols: List[str] = []
        symbol_map: Dict[str, int] = {}
        for token in (tokens or []):
            try:
                t = int(token)
            except Exception:
                continue
            sym = None
            with self._lock:
                sym = self._token_to_symbol.get(t)
            if not sym and self._resolver is not None:
                try:
                    # try resolver.format_token_as_symbol or resolver.symbol_for_token
                    fmt = getattr(self._resolver, "format_token_as_symbol", None)
                    if callable(fmt):
                        sym = fmt(t)
                    else:
                        # try symbol_for_token if present
                        sft = getattr(self._resolver, "symbol_for_token", None)
                        if callable(sft):
                            sym = sft(t)
                except Exception:
                    LOGGER.debug("resolver token->symbol lookup failed", exc_info=True)
            if not sym:
                # try broker cache: some brokers allow asking by token -> symbol via instruments cache
                try:
                    # broker may offer instrument symbol mapping
                    bfn = getattr(self._broker, "get_quote_by_token", None)
                    if callable(bfn):
                        payload = bfn(int(t))
                        if isinstance(payload, Mapping):
                            # determine input key used by broker responses
                            sym = payload.get("instrument_symbol") or payload.get("tradingsymbol") or payload.get("instrument") or None
                except Exception:
                    pass
            if not sym:
                sym = str(t)
            sym_str = str(sym)
            out_symbols.append(sym_str)
            symbol_map[sym_str] = int(t)
            # try to register mapping into local caches if symbol contains token info
            try:
                if isinstance(sym, str) and ":" in sym:
                    # register both 'EX:SYM' and bare 'SYM'
                    self._register_token_symbol(int(t), sym_str)
            except Exception:
                pass
        return out_symbols, symbol_map

    def _register_token_symbol(self, token: int, symbol: str) -> None:
        with self._lock:
            try:
                tok = int(token)
            except Exception:
                return
            if not symbol:
                return
            norm = symbol.strip()
            self._token_to_symbol[tok] = norm
            base = norm.split(":", 1)[-1]
            self._symbol_to_token.setdefault(norm, tok)
            self._symbol_to_token.setdefault(base, tok)

    @staticmethod
    def _extract_token_from_mapping(m: Mapping[str, Any]) -> Optional[int]:
        for k in ("instrument_token", "token", "instrumentToken"):
            v = m.get(k)
            if v is not None:
                try:
                    return int(float(v))
                except Exception:
                    continue
        return None

    # -----------------------
    # Polling thread management
    # -----------------------
    def _start_polling_thread(self) -> None:
        with self._poll_lock:
            if self._is_polling:
                return
            self._stop_event.clear()
            self._is_polling = True
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="mdm-poller")
            self._poll_thread.start()
            LOGGER.info("market_data_poll_thread_started", extra={"event": "market_data_poll_thread_started"})

    def _stop_polling_thread(self) -> None:
        with self._poll_lock:
            if not self._is_polling:
                return
            self._stop_event.set()
            t = self._poll_thread
            self._poll_thread = None
            self._is_polling = False
        if t is not None:
            t.join(timeout=2.0)
        LOGGER.info("market_data_poll_thread_stopped", extra={"event": "market_data_poll_thread_stopped"})

    def _poll_loop(self) -> None:
        """Background poll loop: run until stop_event is set."""
        try:
            while not self._stop_event.is_set():
                tokens: List[int] = []
                with self._poll_lock:
                    tokens = list(self._poll_targets)
                if tokens:
                    try:
                        quote_map = self.get_quote_bulk(tokens)
                        if quote_map:
                            # store snapshot (threadsafe)
                            with self._lock:
                                # map token strings to payloads for quick read access
                                self._last_quote_snapshot = {str(k): v for k, v in quote_map.items() if v is not None}
                    except Exception:
                        LOGGER.exception("market_data_poll_iter_error", extra={"event": "market_data_poll_iter_error"})
                # avoid tight loop; sleep configured interval
                self._stop_event.wait(self._poll_interval)
        except Exception:
            LOGGER.exception("market_data_poll_loop_crashed", extra={"event": "market_data_poll_loop_crash"})
        finally:
            with self._poll_lock:
                self._is_polling = False

    # -----------------------
    # Websocket integration helpers (best-effort)
    # -----------------------
    def _attach_ws_handlers(self, ws: Any) -> None:
        """Wire minimal handlers on websocket manager if it supports callbacks."""
        try:
            # attempt to set on_tick handler if websocket manager expects it
            if hasattr(ws, "on_tick"):
                try:
                    ws.on_tick = self._on_ws_tick  # type: ignore[attr-defined]
                except Exception:
                    # if it's a property, attempt set by setter method or attribute
                    try:
                        setattr(ws, "on_tick", self._on_ws_tick)
                    except Exception:
                        pass
            # If websocket manager expects an error handler
            if hasattr(ws, "on_error"):
                with suppress(Exception):
                    ws.on_error = self._on_ws_error  # type: ignore[attr-defined]
        except Exception:
            LOGGER.debug("attach_ws_handlers failed", exc_info=True)

    def _on_ws_tick(self, tick: Mapping[str, Any]) -> None:
        """Callback for websocket tick payloads. Keep minimal processing here."""
        try:
            if not isinstance(tick, Mapping):
                return
            # If payload includes instrument_token, register mapping
            it = tick.get("instrument_token") or tick.get("token")
            if it:
                try:
                    tok = int(it)
                    # Try to extract a tradingsymbol if present in payload
                    sym = tick.get("tradingsymbol") or tick.get("symbol")
                    if sym:
                        # normalise format
                        if ":" not in sym:
                            sym = f"{self._default_exchange}:{sym}"
                        self._register_token_symbol(tok, sym)
                except Exception:
                    pass
            # store recent tick for read-through diagnostics
            with self._lock:
                key = str(tick.get("instrument_token") or tick.get("token") or tick.get("tradingsymbol") or "")
                if key:
                    self._last_quote_snapshot[key] = dict(tick)
        except Exception:
            LOGGER.exception("ws_tick_handler_error", extra={"event": "market_data_ws_tick_error"})

    def _on_ws_error(self, exc: Exception) -> None:
        LOGGER.warning("market_data_ws_error", extra={"event": "market_data_ws_error", "error": str(exc)})

    # -----------------------
    # Helpers for external tooling
    # -----------------------
    def load_rows_into_resolver(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Warm the attached resolver from CSV/DB rows (caller-supplied)."""
        if not rows or self._resolver is None:
            return
        try:
            if hasattr(self._resolver, "warm_from_broker_dump"):
                self._resolver.warm_from_broker_dump(rows)
            elif hasattr(self._resolver, "warm"):
                # fallback diffing
                self._resolver.warm()
        except Exception:
            LOGGER.exception("load_rows_into_resolver failed", extra={"event": "market_data_load_rows_into_resolver_error"})

    def load_instruments_into_cache(self) -> None:
        """
        If broker exposes list_instruments or load_instruments, load instrument metadata into local caches.
        Safe to call at startup (non-fatal).
        """
        try:
            instruments = []
            li = getattr(self._broker, "list_instruments", None)
            if callable(li):
                with suppress(Exception):
                    instruments = li()
            elif callable(getattr(self._broker, "load_instruments", None)):
                with suppress(Exception):
                    instruments = self._broker.load_instruments()
            if not instruments:
                return
            for ins in instruments:
                try:
                    token = self._extract_token_from_mapping(ins)
                    sym = (ins.get("tradingsymbol") or ins.get("symbol") or "").strip()
                    if token and sym:
                        if ":" not in sym:
                            sym = f"{self._default_exchange}:{sym}"
                        self._register_token_symbol(int(token), sym)
                except Exception:
                    continue
        except Exception:
            LOGGER.exception("load_instruments_into_cache failed", extra={"event": "market_data_load_instruments_error"})

    def stop(self) -> None:
        """Stop polling thread and detach websocket handlers (best-effort)."""
        try:
            self._stop_polling_thread()
        except Exception:
            LOGGER.debug("stop: poll thread stop failed", exc_info=True)
        try:
            # if websocket manager has disconnect/close, call it is not this responsibility normally
            if self._ws is not None and hasattr(self._ws, "disconnect"):
                with suppress(Exception):
                    self._ws.disconnect()
        except Exception:
            LOGGER.debug("stop: ws disconnect failed", exc_info=True)

    # -----------------------
    # Internal minor helpers
    # -----------------------
    def __repr__(self) -> str:
        with self._lock:
            return f"<MarketDataManager ws={'yes' if self._ws else 'no'} resolver={'yes' if self._resolver else 'no'} poll_targets={len(self._poll_targets)}>"

