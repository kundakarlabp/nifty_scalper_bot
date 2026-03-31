# ruff: noqa: I001

"""Unified manager that composes core market data and risk services."""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Optional

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.pricing import canonical_price_source

LOGGER = get_logger(__name__).getChild("unified_manager")


@dataclass(frozen=True)
class UMInstrument:
    """Resolved instrument metadata returned by :class:`UnifiedManager`."""

    symbol: str
    exchange: Optional[str]
    token: Optional[int]
    lot_size: int
    kind: str
    expiry: Optional[str] = None


@dataclass(frozen=True)
class UMPrice:
    """Pricing snapshot returned by :class:`UnifiedManager`."""

    symbol: str
    ltp: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    ts: float
    source: str


@dataclass
class UMPlan:
    """Sizing outcome for :meth:`UnifiedManager.plan`."""

    ok: bool
    qty: int
    est_required: float
    reason: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


def _coerce_pos_float(value: Any) -> Optional[float]:
    """Convert ``value`` to a positive float if possible.

    Args:
        value: Arbitrary value possibly representing a number.

    Returns:
        A positive float if coercion succeeded, otherwise ``None``.

    Raises:
        None.
    """

    try:
        result = float(value)
    except Exception as exc:  # pragma: no cover - defensive conversion
        LOGGER.debug(
            "Condition met: um_coerce_float_failed",
            extra={"event": "um_coerce_float_failed", "error": str(exc)},
        )
        return None
    if result > 0 and math.isfinite(result):
        return result
    return None


class UnifiedManager:
    """Provide a narrow interface for Telegram diagnostics and planning."""

    def __init__(
        self,
        *,
        broker: Any,
        mdm: Any,
        ws: Any | None,
        risk: Any | None,
        streamer: Any | None = None,
        data_hub: Any | None = None,
        orders: Any | None = None,
        strategies: Any | None = None,
        logger=None,
    ) -> None:
        """Create a new :class:`UnifiedManager` instance.

        Args:
            broker: Broker client exposing quote and instrument helpers.
            mdm: Market data manager with cache and polling helpers.
            ws: Optional websocket manager for live subscriptions.
            risk: Risk manager exposing balance/config.
            logger: Optional logger override for fine-grained routing.

        Returns:
            None.

        Raises:
            None.
        """

        self.log = logger or LOGGER
        self.log.debug(
            "Entered UnifiedManager.__init__",
            extra={"event": "um_init_enter"},
        )
        self.broker = broker
        self.mdm = mdm
        self.ws = ws
        self.risk = risk
        self.streamer = streamer
        self.data_hub = data_hub
        self.orders = orders
        self.strategies = strategies
        self._cache_snapshot: dict[str, dict[str, Any]] = {}
        self._last_balance_value: float | None = None
        self._last_balance_source: str = "unknown"
        self.resolver = getattr(mdm, "_resolver", None) or getattr(
            mdm, "resolver", None
        )
        self._feature_order_without_token: bool = True
        self._feature_resolver_learn_from_quotes: bool = True
        try:
            runtime_settings = get_settings()
            self._feature_order_without_token = bool(
                getattr(runtime_settings, "feature_order_without_token", True)
            )
            self._feature_resolver_learn_from_quotes = bool(
                getattr(
                    runtime_settings,
                    "feature_resolver_learn_from_quotes",
                    True,
                )
            )
            self.log.info(
                "Condition met: um_features_loaded",
                extra={
                    "event": "um_features_loaded",
                    "order_without_token": self._feature_order_without_token,
                    "learn_from_quotes": self._feature_resolver_learn_from_quotes,
                },
            )
        except Exception as exc:  # noqa: BLE001 - defensive feature fallback
            self._feature_order_without_token = True
            self._feature_resolver_learn_from_quotes = True
            self.log.error(
                "Failure in UnifiedManager.__init__ settings load: %s",
                exc,
                extra={"event": "um_features_load_error"},
                exc_info=exc,
            )
        self.log.info("Condition met: um_init_ready", extra={"event": "um_init_ready"})

    def on_cache_update(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Record cache updates forwarded by collaborators.

        Args:
            symbol: Normalized instrument identifier for the tick payload.
            tick: Mapping containing normalized tick fields.

        Returns:
            None.

        Raises:
            None.
        """

        if not symbol:
            return
        snapshot_tick = dict(tick)
        if symbol not in self._cache_snapshot:
            self.log.debug(
                "Entered UnifiedManager.on_cache_update",
                extra={"event": "um_cache_update_enter", "symbol": symbol},
            )
        self._cache_snapshot[symbol] = snapshot_tick

    def on_balance_update(self, balance: float, *, source: str = "live") -> None:
        """Record asynchronous balance updates for later use.

        Args:
            balance: Latest account balance figure.
            source: Descriptive label for the balance origin.

        Returns:
            None.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.on_balance_update",
            extra={"event": "um_balance_update_enter", "source": source},
        )
        try:
            self._last_balance_value = float(balance)
            self._last_balance_source = str(source or "unknown")
        except Exception as exc:  # noqa: BLE001 - defensive cast
            self.log.error(
                "Failure in UnifiedManager.on_balance_update: %s",
                exc,
                extra={"event": "um_balance_update_error"},
                exc_info=exc,
            )

    def resolve(self, symbol: str) -> UMInstrument:
        """Resolve instrument metadata for ``symbol``.

        Args:
            symbol: Instrument identifier in NSE/NFO namespace.

        Returns:
            A populated :class:`UMInstrument` with available metadata.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.resolve",
            extra={"event": "um_resolve_enter", "symbol": symbol},
        )
        try:
            exchange: Optional[str] = None
            token: Optional[int] = None
            lot_size = 1
            kind = "EQ"
            expiry: Optional[str] = None

            resolver_fn = getattr(self.mdm, "resolve_symbol_token", None)
            if callable(resolver_fn):
                try:
                    resolved_token = resolver_fn(symbol)
                    if resolved_token is not None:
                        token = int(resolved_token)
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.error(
                        "Failure in UnifiedManager.resolve resolver_fn: %s",
                        exc,
                        extra={"event": "um_resolve_token_error", "symbol": symbol},
                        exc_info=exc,
                    )
            if token is None and self.resolver is not None:
                resolver_obj = self.resolver
                if hasattr(resolver_obj, "resolve_symbol_to_token"):
                    try:
                        resolved_token = resolver_obj.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
                        if resolved_token is not None:
                            token = int(resolved_token)
                            self.log.info(
                                "Condition met: um_resolve_token_resolver",
                                extra={
                                    "event": "um_resolve_token_resolver",
                                    "symbol": symbol,
                                    "token": token,
                                },
                            )
                    except Exception as exc:  # noqa: BLE001
                        self.log.error(
                            "Failure in UnifiedManager.resolve resolver bridge: %s",
                            exc,
                            extra={
                                "event": "um_resolve_token_resolver_error",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
            if token is None:
                try:
                    token_map = getattr(self.mdm, "_token_by_symbol", {}) or {}
                    cached_token = token_map.get(symbol)
                    if cached_token is None:
                        cached_token = token_map.get(symbol.strip().upper())
                    if cached_token is not None:
                        token = int(cached_token)
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.error(
                        "Failure in UnifiedManager.resolve cache lookup: %s",
                        exc,
                        extra={"event": "um_resolve_cache_error", "symbol": symbol},
                        exc_info=exc,
                    )
            with suppress(Exception):
                meta_accessor = getattr(self.mdm, "instrument_meta", None)
                if callable(meta_accessor):
                    meta = meta_accessor(symbol)
                    if isinstance(meta, Mapping):
                        exchange = meta.get("exchange") or exchange
                        lot_size = int(meta.get("lot_size") or lot_size)
                        kind = str(meta.get("kind") or kind)
                        expiry = meta.get("expiry") or expiry

            if token is None and hasattr(self.broker, "resolve_candidates"):
                try:
                    candidates = self._candidate_keys(symbol)
                    resolved = self.broker.resolve_candidates(candidates)
                    if isinstance(resolved, Mapping):
                        token = resolved.get("token") or token
                        exchange = resolved.get("exchange") or exchange
                        lot_size = int(resolved.get("lot_size") or lot_size)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self.log.error(
                        "Failure in UnifiedManager.resolve broker fallback: %s",
                        exc,
                        extra={
                            "event": "um_resolver_broker_fallback_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )

            if exchange is None and token is not None:
                exchange_lookup = getattr(self.mdm, "_exchange_by_token", None)
                if isinstance(exchange_lookup, dict):
                    exchange = exchange_lookup.get(token)
            if exchange is None and self.resolver is not None:
                resolver_obj = self.resolver
                if hasattr(resolver_obj, "resolve_exchange"):
                    try:
                        exchange_hint = resolver_obj.resolve_exchange(symbol)  # type: ignore[attr-defined]
                        if exchange_hint:
                            exchange = exchange_hint
                    except Exception as exc:  # noqa: BLE001
                        self.log.error(
                            "Failure in UnifiedManager.resolve exchange bridge: %s",
                            exc,
                            extra={
                                "event": "um_resolve_exchange_resolver_error",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )

            if lot_size <= 0:
                lot_size = 1
            if self.resolver is not None and lot_size <= 1:
                resolver_obj = self.resolver
                if hasattr(resolver_obj, "lot_size_for_symbol"):
                    try:
                        lot_candidate = resolver_obj.lot_size_for_symbol(symbol)  # type: ignore[attr-defined]
                        if isinstance(lot_candidate, int) and lot_candidate > 0:
                            lot_size = lot_candidate
                    except Exception as exc:  # noqa: BLE001
                        self.log.error(
                            "Failure in UnifiedManager.resolve lot size bridge: %s",
                            exc,
                            extra={
                                "event": "um_resolve_lot_resolver_error",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )

            lot_size = max(lot_size, 1)
            instrument = UMInstrument(
                symbol=symbol,
                exchange=exchange,
                token=token,
                lot_size=lot_size,
                kind=kind,
                expiry=expiry,
            )
            if token is None:
                METRICS.record_resolver_miss(symbol=symbol)
            self.log.info(
                "Condition met: um_resolve_ready",
                extra={"event": "um_resolve_ready", "symbol": symbol, "token": token},
            )
            return instrument
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.resolve: %s",
                exc,
                extra={"event": "um_resolve_error", "symbol": symbol},
                exc_info=exc,
            )
            return UMInstrument(
                symbol=symbol,
                exchange=None,
                token=None,
                lot_size=1,
                kind="EQ",
            )

    def _candidate_keys(self, symbol: str) -> list[str | int]:
        """Build broker lookup keys for ``symbol``.

        Args:
            symbol: User-facing identifier passed to the command.

        Returns:
            Ordered list of lookup keys to attempt.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager._candidate_keys",
            extra={"event": "um_candidates_enter", "symbol": symbol},
        )
        try:
            keys: list[str | int] = []
            norm_symbol = symbol.strip().upper()
            candidate_fn = getattr(self.mdm, "_candidate_quote_keys", None)
            if callable(candidate_fn):
                try:
                    mdm_candidates = candidate_fn(symbol)
                    for candidate in mdm_candidates or []:
                        if candidate in {"", None} or candidate in keys:
                            continue
                        keys.append(candidate)
                except Exception as exc:  # noqa: BLE001 - align with diagnostics
                    self.log.error(
                        "Failure in UnifiedManager._candidate_keys mdm: %s",
                        exc,
                        extra={"event": "um_candidates_mdm_error", "symbol": symbol},
                        exc_info=exc,
                    )
            resolver_obj = self.resolver
            if resolver_obj is not None and hasattr(resolver_obj, "build_quote_keys"):
                try:
                    _canon, resolver_candidates = resolver_obj.build_quote_keys(symbol)  # type: ignore[attr-defined]
                    for candidate in resolver_candidates:
                        if candidate in {"", None} or candidate in keys:
                            continue
                        keys.append(candidate)
                except Exception as exc:  # noqa: BLE001
                    self.log.error(
                        "Failure in UnifiedManager._candidate_keys resolver: %s",
                        exc,
                        extra={
                            "event": "um_candidates_resolver_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
            if not keys:
                if ":" in norm_symbol:
                    keys.append(norm_symbol)
                keys.extend(
                    [
                        f"NSE:{norm_symbol}",
                        norm_symbol,
                        f"NSE:{norm_symbol} 50",
                        f"{norm_symbol} 50",
                    ]
                )
                if any(tag in norm_symbol for tag in ("CE", "PE", "FUT", "OPT")):
                    keys.insert(0, f"NFO:{norm_symbol}")
            resolver_fn = getattr(self.mdm, "resolve_symbol_token", None)
            if callable(resolver_fn):
                try:
                    resolved_token = resolver_fn(symbol)
                    if resolved_token is not None:
                        token_int = int(resolved_token)
                        if token_int not in keys:
                            keys.append(token_int)
                except Exception as exc:  # noqa: BLE001 - defensive
                    self.log.error(
                        "Failure in UnifiedManager._candidate_keys token append: %s",
                        exc,
                        extra={"event": "um_candidates_token_error", "symbol": symbol},
                        exc_info=exc,
                    )
            resolver_obj = self.resolver
            if resolver_obj is not None and hasattr(
                resolver_obj, "resolve_symbol_to_token"
            ):
                try:
                    resolved_token = resolver_obj.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
                    if resolved_token is not None:
                        token_int = int(resolved_token)
                        if token_int not in keys:
                            keys.append(token_int)
                except Exception as exc:  # noqa: BLE001
                    self.log.error(
                        "Failure in UnifiedManager._candidate_keys resolver token: %s",
                        exc,
                        extra={
                            "event": "um_candidates_resolver_token_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
            self.log.info(
                "Condition met: um_candidates_ready",
                extra={"event": "um_candidates_ready", "symbol": symbol, "keys": keys},
            )
            return keys
        except Exception as exc:  # pragma: no cover - defensive
            self.log.error(
                "Failure in UnifiedManager._candidate_keys: %s",
                exc,
                extra={"event": "um_candidates_error", "symbol": symbol},
                exc_info=exc,
            )
            return [symbol.strip().upper()]

    def ensure_tracking(self, symbol: str) -> dict[str, Any]:
        """Ensure websocket or polling coverage for ``symbol``.

        Args:
            symbol: Instrument identifier to monitor.

        Returns:
            Tracking status flags for websocket and polling.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.ensure_tracking",
            extra={"event": "um_ensure_tracking_enter", "symbol": symbol},
        )
        try:
            instrument = self.resolve(symbol)
            subscribed = False
            polled = False
            if instrument.token and self.ws is not None:
                with suppress(Exception):
                    self.ws.subscribe_tokens({int(instrument.token)})
                    subscribed = True
            with suppress(Exception):
                poll = getattr(self.mdm, "enable_poll_tracking", None)
                if callable(poll):
                    poll(symbol)
                    polled = True
            result = {
                "symbol": symbol,
                "token": instrument.token,
                "ws": subscribed,
                "poll": polled,
            }
            self.log.info(
                "Condition met: um_ensure_tracking_ready",
                extra={"event": "um_ensure_tracking_ready", **result},
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.ensure_tracking: %s",
                exc,
                extra={"event": "um_ensure_tracking_error", "symbol": symbol},
                exc_info=exc,
            )
            return {"symbol": symbol, "token": None, "ws": False, "poll": False}

    def transport_status(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Summarise websocket/poll transport state.

        Args:
            symbol: Optional symbol filter for cache rows.

        Returns:
            Transport metrics including websocket, polling, and cache rows.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.transport_status",
            extra={"event": "um_transport_enter", "symbol": symbol},
        )
        try:
            ws_connected = False
            polling = False
            poll_info: dict[str, Any] = {
                "interval": None,
                "max_symbols": None,
                "tokens": None,
            }
            mdm_transport: dict[str, Any] = {}
            transport_fn = getattr(self.mdm, "transport_status", None)
            if callable(transport_fn):
                try:
                    raw_status = transport_fn()
                    if isinstance(raw_status, Mapping):
                        mdm_transport = dict(raw_status)
                except Exception as exc:  # noqa: BLE001 - defensive log surface
                    self.log.error(
                        "Failure in UnifiedManager.transport_status mdm: %s",
                        exc,
                        extra={"event": "um_transport_mdm_error", "symbol": symbol},
                        exc_info=exc,
                    )
            if mdm_transport:
                ws_connected = bool(mdm_transport.get("ws_connected", False))
                polling = bool(
                    mdm_transport.get(
                        "poll_enabled", mdm_transport.get("polling", False)
                    )
                )
                poll_info["interval"] = mdm_transport.get("poll_interval")
                poll_info["max_symbols"] = mdm_transport.get("poll_max")
                if "poll_tokens" in mdm_transport:
                    poll_info["tokens"] = mdm_transport.get("poll_tokens")
            else:
                try:
                    ws_connected = bool(self.ws and self.ws.is_connected())
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.debug(
                        "Failure in UnifiedManager.transport_status ws fallback: %s",
                        exc,
                        extra={
                            "event": "um_transport_ws_fallback_error",
                            "symbol": symbol,
                        },
                    )
                streamer = self.streamer
                if streamer is not None:
                    try:
                        polling = bool(streamer.is_running())
                    except Exception as exc:  # noqa: BLE001 - defensive guard
                        self.log.debug(
                            "UM.transport_status streamer state failed: %s",
                            exc,
                            extra={
                                "event": "um_transport_streamer_state_error",
                                "symbol": symbol,
                            },
                        )
                    try:
                        poll_info["interval"] = poll_info.get("interval") or getattr(
                            streamer, "_interval", None
                        )
                        symbols_attr = getattr(streamer, "_symbols", None)
                        if isinstance(symbols_attr, (set, list, tuple)):
                            poll_info["tokens"] = len(symbols_attr)
                        elif isinstance(symbols_attr, Mapping):
                            poll_info["tokens"] = len(symbols_attr)
                        poll_info["batch_size"] = getattr(
                            streamer, "_batch_size", poll_info.get("batch_size")
                        )
                    except Exception as exc:  # noqa: BLE001 - defensive guard
                        self.log.debug(
                            "UM.transport_status streamer meta failed: %s",
                            exc,
                            extra={
                                "event": "um_transport_streamer_meta_error",
                                "symbol": symbol,
                            },
                        )
                try:
                    poll_info["interval"] = poll_info.get("interval") or getattr(
                        self.mdm, "_rest_poll_interval", None
                    )
                    poll_info["max_symbols"] = getattr(
                        self.mdm, "_rest_poll_max_symbols", None
                    )
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.debug(
                        "Failure in UnifiedManager.transport_status poll fallback: %s",
                        exc,
                        extra={
                            "event": "um_transport_poll_fallback_error",
                            "symbol": symbol,
                        },
                    )
            cache_rows: list[dict[str, Any]] = []
            latest_ticks: Mapping[str, Mapping[str, Any]] = {}
            try:
                latest_ticks = dict(getattr(self.mdm, "_latest_ticks", {}))
            except Exception as exc:  # noqa: BLE001 - defensive guard
                self.log.debug(
                    "Failure in UnifiedManager.transport_status latest cache: %s",
                    exc,
                    extra={"event": "um_transport_cache_error", "symbol": symbol},
                )
            if not latest_ticks:
                latest_ticks = dict(self._cache_snapshot)
            now = time.time()
            for sym, tick in latest_ticks.items():
                if symbol and sym != symbol:
                    continue
                age = None
                ts_value = tick.get("timestamp")
                if ts_value:
                    try:
                        age = max(0.0, now - float(ts_value))
                    except Exception as exc:  # noqa: BLE001 - defensive cast
                        self.log.debug(
                            "Failure in UnifiedManager.transport_status age calc: %s",
                            exc,
                            extra={"event": "um_transport_age_error", "symbol": sym},
                        )
                token_value = None
                resolver_fn = getattr(self.mdm, "resolve_symbol_token", None)
                if callable(resolver_fn):
                    try:
                        token_candidate = resolver_fn(sym)
                        if token_candidate is not None:
                            token_value = int(token_candidate)
                    except Exception as exc:  # noqa: BLE001 - defensive guard
                        self.log.debug(
                            "UM.transport_status token resolve failed: %s",
                            exc,
                            extra={
                                "event": "um_transport_token_resolve_error",
                                "symbol": sym,
                            },
                        )
                if token_value is None:
                    try:
                        token_map = getattr(self.mdm, "_token_by_symbol", {})
                        token_raw = token_map.get(sym)
                        if token_raw is not None:
                            token_value = int(token_raw)
                    except Exception as exc:  # noqa: BLE001 - defensive guard
                        self.log.debug(
                            "UM.transport_status token cache failed: %s",
                            exc,
                            extra={
                                "event": "um_transport_token_cache_error",
                                "symbol": sym,
                            },
                        )
                snapshot_tick = dict(tick)
                cache_rows.append(
                    {
                        "symbol": sym,
                        "ltp": tick.get("ltp"),
                        "bid": tick.get("bid"),
                        "ask": tick.get("ask"),
                        "age": age,
                        "src": tick.get("_src") or "rest",
                        "token": token_value,
                    }
                )
                self._cache_snapshot[sym] = snapshot_tick
            result = {
                "ws_connected": ws_connected,
                "polling": polling,
                "poll": poll_info,
                "cache": cache_rows,
                "cache_symbols": len(cache_rows),
            }
            self.log.info(
                "Condition met: um_transport_ready",
                extra={
                    "event": "um_transport_ready",
                    "symbol": symbol,
                    "rows": len(cache_rows),
                },
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.transport_status: %s",
                exc,
                extra={"event": "um_transport_error", "symbol": symbol},
                exc_info=exc,
            )
            return {"ws_connected": False, "polling": False, "poll": {}, "cache": []}

    def get_price(self, symbol: str) -> UMPrice:
        """Return consolidated price snapshot for ``symbol``.

        Args:
            symbol: Instrument identifier to price.

        Returns:
            A :class:`UMPrice` representing the best-known snapshot.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.get_price",
            extra={"event": "um_price_enter", "symbol": symbol},
        )
        now = time.time()
        try:
            tick: Mapping[str, Any] | None = None
            latest_fn = getattr(self.mdm, "get_latest_tick", None)
            if callable(latest_fn):
                try:
                    latest_tick = latest_fn(symbol)
                    if isinstance(latest_tick, Mapping):
                        tick = latest_tick
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.error(
                        "Failure in UnifiedManager.get_price latest tick: %s",
                        exc,
                        extra={"event": "um_price_latest_error", "symbol": symbol},
                        exc_info=exc,
                    )
            if tick is None:
                try:
                    cache_map = getattr(self.mdm, "_latest_ticks", {})
                    cached_tick = cache_map.get(symbol)
                    if isinstance(cached_tick, Mapping):
                        tick = cached_tick
                except Exception as exc:  # noqa: BLE001 - defensive guard
                    self.log.debug(
                        "Failure in UnifiedManager.get_price cache lookup: %s",
                        exc,
                        extra={
                            "event": "um_price_cache_lookup_error",
                            "symbol": symbol,
                        },
                    )
            if isinstance(tick, Mapping):
                ltp = _coerce_pos_float(tick.get("ltp"))
                if ltp is not None:
                    price = UMPrice(
                        symbol=symbol,
                        ltp=ltp,
                        bid=_coerce_pos_float(tick.get("bid")),
                        ask=_coerce_pos_float(tick.get("ask")),
                        ts=float(tick.get("timestamp") or now),
                        source="mdm",
                    )
                    self.log.info(
                        "Condition met: um_price_cache_hit",
                        extra={"event": "um_price_cache_hit", "symbol": symbol},
                    )
                    return price
            quote = None
            try:
                keys = self._candidate_keys(symbol)
                if hasattr(self.broker, "quote"):
                    quote = self.broker.quote(keys)
                elif hasattr(self.broker, "get_quote"):
                    quote = self.broker.get_quote(keys)
            except Exception as exc:  # pragma: no cover - defensive
                self.log.error(
                    "Failure in UnifiedManager.get_price broker call: %s",
                    exc,
                    extra={"event": "um_broker_quote_failed", "symbol": symbol},
                    exc_info=exc,
                )
            normalized = self._normalize_quote(symbol, quote, now)
            if normalized.ltp is not None:
                if self._feature_resolver_learn_from_quotes:
                    self._maybe_learn_from_quote(symbol, quote)
                try:
                    if hasattr(self.mdm, "ingest_rest_quote"):
                        rest_payload = {
                            "ltp": normalized.ltp,
                            "bid": normalized.bid,
                            "ask": normalized.ask,
                            "timestamp": normalized.ts,
                        }
                        self.mdm.ingest_rest_quote(symbol, rest_payload)
                except Exception as exc:  # noqa: BLE001 - defensive bridge
                    self.log.error(
                        "Failure UM→MDM cache commit: %s",
                        exc,
                        extra={
                            "event": "um_commit_cache_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                ingestor = getattr(self.mdm, "ingest_manual", None)
                if callable(ingestor):
                    try:
                        ingestor(
                            symbol,
                            normalized.ltp,
                            normalized.bid,
                            normalized.ask,
                            normalized.ts,
                            src="seed",
                        )
                    except Exception as exc:  # noqa: BLE001 - defensive guard
                        self.log.debug(
                            "Failure in UnifiedManager.get_price ingest_manual: %s",
                            exc,
                            extra={
                                "event": "um_price_ingest_manual_error",
                                "symbol": symbol,
                            },
                        )
                self.log.info(
                    "Condition met: um_price_broker_hit",
                    extra={"event": "um_price_broker_hit", "symbol": symbol},
                )
                return normalized
            self.log.info(
                "Condition met: um_price_empty",
                extra={"event": "um_price_empty", "symbol": symbol},
            )
            return UMPrice(symbol, None, None, None, now, "mdm-empty")
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.get_price: %s",
                exc,
                extra={"event": "um_price_error", "symbol": symbol},
                exc_info=exc,
            )
            return UMPrice(symbol, None, None, None, now, "error")

    def _normalize_quote(self, symbol: str, raw: Any, default_ts: float) -> UMPrice:
        """Normalize broker payloads into :class:`UMPrice`.

        Args:
            symbol: Instrument identifier being normalised.
            raw: Broker payload from quote/get_quote.
            default_ts: Timestamp to fallback when payload lacks one.

        Returns:
            Normalised :class:`UMPrice` instance.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager._normalize_quote",
            extra={"event": "um_normalize_enter", "symbol": symbol},
        )
        try:
            if not raw:
                return UMPrice(symbol, None, None, None, default_ts, "broker")
            payload: Mapping[str, Any] | None = None
            if isinstance(raw, Mapping):
                payload = raw
            elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], Mapping):
                payload = raw[0]
            else:
                payload = None

            def first_numeric(*values: Any) -> Optional[float]:
                for candidate in values:
                    coerced = _coerce_pos_float(candidate)
                    if coerced is not None:
                        return coerced
                return None

            timestamp = default_ts
            if payload is not None:
                with suppress(Exception):
                    timestamp = float(payload.get("timestamp") or default_ts)
                ltp = first_numeric(
                    payload.get("last_price"),
                    payload.get("ltp"),
                    payload.get("price"),
                    payload.get("close"),
                )
                bid = first_numeric(payload.get("buy_price"), payload.get("bid"))
                ask = first_numeric(payload.get("sell_price"), payload.get("ask"))
            else:
                ltp = bid = ask = None
            result = UMPrice(symbol, ltp, bid, ask, timestamp, "broker")
            self.log.info(
                "Condition met: um_normalize_ready",
                extra={"event": "um_normalize_ready", "symbol": symbol},
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager._normalize_quote: %s",
                exc,
                extra={"event": "um_normalize_error", "symbol": symbol},
                exc_info=exc,
            )
            return UMPrice(symbol, None, None, None, default_ts, "broker")

    def plan(self, symbol: str, side: str, qty: int) -> UMPlan:
        """Compute trade sizing for ``symbol``.

        Args:
            symbol: Instrument identifier to plan for.
            side: Trade direction (``BUY`` or ``SELL``).
            qty: Requested number of lots/contracts.

        Returns:
            A :class:`UMPlan` describing sizing or rejection reason.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.plan",
            extra={
                "event": "um_plan_enter",
                "symbol": symbol,
                "side": side,
                "qty": qty,
            },
        )
        try:
            side_norm = side.strip().upper()
            if side_norm not in {"BUY", "SELL"}:
                self.log.info(
                    "Condition met: um_plan_bad_side",
                    extra={"event": "um_plan_bad_side", "side": side},
                )
                return UMPlan(False, 0, 0.0, "BAD_SIDE", {"hint": "Use BUY/SELL"})

            instrument = self.resolve(symbol)
            resolver_obj = self.resolver
            tradingsymbol: str | None = None
            exchange: str | None = None
            if resolver_obj is not None:
                if hasattr(resolver_obj, "tradingsymbol_for_order"):
                    try:
                        tradingsymbol = resolver_obj.tradingsymbol_for_order(symbol)  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001
                        self.log.debug(
                            "um_plan_tradingsymbol_resolver_failed",
                            extra={
                                "event": "um_plan_tradingsymbol_resolver_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
                if hasattr(resolver_obj, "exchange_for_symbol"):
                    try:
                        exchange = resolver_obj.exchange_for_symbol(symbol)  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001
                        self.log.debug(
                            "um_plan_exchange_resolver_failed",
                            extra={
                                "event": "um_plan_exchange_resolver_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
            if exchange is None and instrument.exchange:
                exchange = instrument.exchange
            if (
                not self._feature_order_without_token
                and instrument.token is None
                and instrument.kind != "EQ"
            ):
                self.log.info(
                    "Condition met: um_plan_no_token",
                    extra={"event": "um_plan_no_token", "symbol": symbol},
                )
                return UMPlan(False, 0, 0.0, "NO_TOKEN", {"hint": "resolver"})
            if self._feature_order_without_token and instrument.kind != "EQ":
                if not tradingsymbol or not exchange:
                    self.log.info(
                        "Condition met: um_plan_missing_resolution",
                        extra={
                            "event": "um_plan_missing_resolution",
                            "symbol": symbol,
                            "tradingsymbol": tradingsymbol,
                            "exchange": exchange,
                        },
                    )
                    return UMPlan(
                        False,
                        0,
                        0.0,
                        "NO_RESOLUTION",
                        {"hint": "resolver"},
                    )

            price = self.get_price(instrument.symbol)
            if price.ltp is None or price.ltp <= 0:
                METRICS.record_plan_no_price(symbol=symbol)
                self.log.info(
                    "Condition met: um_plan_no_price",
                    extra={"event": "um_plan_no_price", "symbol": symbol},
                )
                return UMPlan(False, 0, 0.0, "NO_PRICE", {"hint": "ingestion"})

            lots = max(1, int(qty))
            lot_size = max(1, int(instrument.lot_size))
            est_required = float(price.ltp) * float(lot_size) * float(lots)

            balance = None
            cap_pct = 100.0
            per_trade_risk = 100.0
            if self.risk is not None:
                with suppress(Exception):
                    balance = float(getattr(self.risk, "current_balance"))
                with suppress(Exception):
                    settings = getattr(self.risk, "settings", None)
                    if settings is not None:
                        cap_pct = float(getattr(settings, "per_trade_cap_pct", cap_pct))
                        per_trade_risk = float(
                            getattr(settings, "per_trade_risk_pct", per_trade_risk)
                        )
            max_capital = (balance or 0.0) * (cap_pct / 100.0)
            if max_capital > 0 and est_required > max_capital:
                max_lots = int(max_capital // (price.ltp * lot_size))
                if max_lots <= 0:
                    self.log.info(
                        "Condition met: um_plan_margin_block",
                        extra={
                            "event": "um_plan_margin_block",
                            "symbol": symbol,
                            "required": est_required,
                        },
                    )
                    return UMPlan(
                        False,
                        0,
                        est_required,
                        "MARGIN",
                        {"hint": "Reduce quantity"},
                    )
                lots = max_lots
                est_required = float(price.ltp) * float(lot_size) * float(lots)
            if est_required <= 0:
                METRICS.record_plan_estimate_zero(symbol=symbol)
                self.log.info(
                    "Condition met: um_plan_estimate_zero",
                    extra={
                        "event": "um_plan_estimate_zero",
                        "symbol": symbol,
                        "lot_size": lot_size,
                        "qty": lots,
                    },
                )

            meta = {
                "price_source": canonical_price_source(price.source),
                "ltp": price.ltp,
                "lot_size": lot_size,
                "per_trade_risk_pct": per_trade_risk,
            }
            if tradingsymbol:
                meta["tradingsymbol"] = tradingsymbol
            if exchange:
                meta["exchange"] = exchange
            plan_result = UMPlan(True, lots, est_required, None, meta)
            self.log.info(
                "Condition met: um_plan_ready",
                extra={
                    "event": "um_plan_ready",
                    "symbol": symbol,
                    "qty": lots,
                    "est_required": est_required,
                },
            )
            return plan_result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.plan: %s",
                exc,
                extra={"event": "um_plan_error", "symbol": symbol},
                exc_info=exc,
            )
            return UMPlan(False, 0, 0.0, "ERROR", {"hint": "internal"})

    def _maybe_learn_from_quote(self, symbol: str, quote: Any) -> None:
        """Best-effort resolver learning from broker quote payloads.

        Args:
            symbol: Instrument identifier associated with the quote.
            quote: Raw broker payload returned by the quote endpoint.

        Returns:
            None.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager._maybe_learn_from_quote",
            extra={"event": "um_learn_enter", "symbol": symbol},
        )
        resolver_obj = self.resolver
        if resolver_obj is None or not hasattr(resolver_obj, "_ingest_instrument_row"):
            return
        payload: Mapping[str, Any] | None = None
        if isinstance(quote, Mapping):
            payload = quote
        elif (
            isinstance(quote, (list, tuple)) and quote and isinstance(quote[0], Mapping)
        ):
            payload = quote[0]
        if payload is None:
            return
        token_value = payload.get("instrument_token") or payload.get("token")
        if token_value is None:
            return
        try:
            token_int = int(float(token_value))
        except (TypeError, ValueError) as exc:
            self.log.debug(
                "um_learn_token_cast_failed",
                extra={
                    "event": "um_learn_token_cast_failed",
                    "symbol": symbol,
                    "token_value": token_value,
                },
                exc_info=exc,
            )
            return
        try:
            existing_token = None
            if hasattr(resolver_obj, "resolve_symbol_to_token"):
                existing_token = resolver_obj.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
            if existing_token is not None and int(existing_token) == token_int:
                return
        except Exception as exc:  # noqa: BLE001
            self.log.debug(
                "um_learn_existing_token_lookup_failed",
                extra={
                    "event": "um_learn_existing_token_lookup_failed",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
        ingest_payload = {
            "instrument_token": token_int,
            "tradingsymbol": payload.get("tradingsymbol")
            or payload.get("symbol")
            or symbol,
            "exchange": payload.get("exchange")
            or payload.get("exchange_segment")
            or payload.get("exch"),
        }
        try:
            resolver_obj._ingest_instrument_row(ingest_payload)  # type: ignore[attr-defined]
            self.log.info(
                "Condition met: um_learn_token_ingested",
                extra={
                    "event": "um_learn_token_ingested",
                    "symbol": symbol,
                    "token": token_int,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.log.debug(
                "um_learn_ingest_failed",
                extra={
                    "event": "um_learn_ingest_failed",
                    "symbol": symbol,
                    "token": token_int,
                },
                exc_info=exc,
            )

    def quote_probe(self, symbol: str) -> dict[str, Any]:
        """Run a quote ingestion probe for ``symbol``.

        Args:
            symbol: Instrument identifier to probe.

        Returns:
            Structured dictionary detailing resolver, price, and cache state.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.quote_probe",
            extra={"event": "um_probe_enter", "symbol": symbol},
        )
        try:
            instrument = self.resolve(symbol)
            price = self.get_price(instrument.symbol)
            transport = self.transport_status(symbol=instrument.symbol)
            candidate_keys = self._candidate_keys(symbol)
            result = {
                "resolver": {
                    "symbol": instrument.symbol,
                    "exchange": instrument.exchange,
                    "token": instrument.token,
                    "lot_size": instrument.lot_size,
                    "kind": instrument.kind,
                    "expiry": instrument.expiry,
                    "candidates": candidate_keys,
                },
                "normalize": {
                    "ltp": price.ltp,
                    "bid": price.bid,
                    "ask": price.ask,
                    "ts": price.ts,
                    "source": canonical_price_source(price.source),
                },
                "transport": {
                    "ws_connected": transport["ws_connected"],
                    "polling": transport["polling"],
                    "poll": transport["poll"],
                },
                "cache": transport["cache"],
            }
            self.log.info(
                "Condition met: um_probe_ready",
                extra={"event": "um_probe_ready", "symbol": symbol},
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.quote_probe: %s",
                exc,
                extra={"event": "um_probe_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "resolver": {},
                "normalize": {
                    "ltp": None,
                    "bid": None,
                    "ask": None,
                    "ts": time.time(),
                    "source": "error",
                },
                "transport": {"ws_connected": False, "polling": False, "poll": {}},
                "cache": [],
            }

    def status(self) -> dict[str, Any]:
        """Return high-level manager status summary.

        Args:
            None.

        Returns:
            Dictionary including websocket, polling, and balance data.

        Raises:
            None.
        """

        self.log.debug(
            "Entered UnifiedManager.status",
            extra={"event": "um_status_enter"},
        )
        try:
            transport = self.transport_status()
            ws_connected = bool(transport.get("ws_connected"))
            poll_info = transport.get("poll", {})
            polling_state: dict[str, Any] = {
                "on": bool(transport.get("polling")),
                "tokens": poll_info.get("tokens", 0),
                "interval": poll_info.get("interval"),
            }
            if "batch_size" in poll_info:
                polling_state["batch_size"] = poll_info.get("batch_size")
            cache_symbols = transport.get("cache_symbols")
            if cache_symbols is None:
                cache_symbols = len(transport.get("cache", []) or [])
            balance = None
            balance_source = "unknown"
            if self.risk is not None:
                try:
                    balance = float(self.risk.refresh_account_balance(force=False))
                    self._last_balance_value = balance
                except Exception:  # noqa: BLE001 - fallback to cached value
                    balance = None
                try:
                    balance_source = str(self.risk.balance_source_label())
                    self._last_balance_source = balance_source
                except Exception:  # noqa: BLE001 - fallback to cached label
                    balance_source = self._last_balance_source
            if balance is None:
                balance = self._last_balance_value
                balance_source = self._last_balance_source
            result = {
                "ws_connected": ws_connected,
                "polling": polling_state,
                "symbols": cache_symbols,
                "cache_symbols": cache_symbols,
                "balance": balance,
                "balance_source": balance_source,
            }
            self.log.info(
                "Condition met: um_status_ready",
                extra={"event": "um_status_ready", **result},
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            self.log.error(
                "Failure in UnifiedManager.status: %s",
                exc,
                extra={"event": "um_status_error"},
                exc_info=exc,
            )
            return {
                "ws_connected": False,
                "polling": False,
                "symbols": 0,
                "balance": None,
                "balance_source": "error",
            }
