"""Runtime hardening for CandleEngine state reconciliation.

This module installs narrow, idempotent wrappers around the existing candle
engine.  It deliberately leaves strategy, order, subscription, and historical
fetch policy unchanged while enforcing the closed-minute watermark invariant:

    current_candle is None or current_minute > latest_completed_minute

The wrappers also serialize history replacement, tick processing, and clock
flush for each engine instance.  This prevents bootstrap/history synchronization
and delayed WebSocket ticks from reconstructing an already-finalized minute.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.data.time_contract import normalize_market_tick_timestamp
from nifty_scalper_bot.utils.logging import log_throttled

LOGGER = logging.getLogger(__name__)
_INSTALLED_ATTR = "_candle_state_hardening_installed"
_ORIGINAL_INIT_ATTR = "_candle_state_hardening_original_init"
_ORIGINAL_REPLACE_ATTR = "_candle_state_hardening_original_replace_history"
_ORIGINAL_ON_TICK_ATTR = "_candle_state_hardening_original_on_tick"
_ORIGINAL_FINALIZE_ATTR = "_candle_state_hardening_original_finalize"
_ORIGINAL_FLUSH_ATTR = "_candle_state_hardening_original_flush"
_ORIGINAL_DIAGNOSTICS_ATTR = "_candle_state_hardening_original_diagnostics"

# CandleEngine is a slotted dataclass without a weak-reference slot, so runtime
# synchronization state cannot safely be attached to the instance. Engine count
# is bounded by the subscribed universe; entries therefore remain bounded by the
# process's engine population.
_REGISTRY_GUARD = threading.RLock()
_ENGINE_LOCKS: dict[int, threading.RLock] = {}
_COUNTERS: dict[int, defaultdict[str, int]] = {}


def _lock_for(engine: Any) -> threading.RLock:
    key = id(engine)
    with _REGISTRY_GUARD:
        lock = _ENGINE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ENGINE_LOCKS[key] = lock
        _COUNTERS.setdefault(key, defaultdict(int))
        return lock


def _counters_for(engine: Any) -> defaultdict[str, int]:
    _lock_for(engine)
    return _COUNTERS[id(engine)]


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def _latest_completed_minute(engine: Any) -> pd.Timestamp:
    completed = getattr(engine, "_completed_candles", None)
    if not completed:
        return pd.NaT
    try:
        return _coerce_timestamp(completed[-1].get("timestamp"))
    except Exception:
        return pd.NaT


def _current_minute(engine: Any) -> pd.Timestamp:
    current = getattr(engine, "current_candle", None)
    if not isinstance(current, Mapping):
        return pd.NaT
    return _coerce_timestamp(current.get("timestamp"))


def _state_consistent(engine: Any) -> bool:
    completed = getattr(engine, "_completed_candles", None) or ()
    previous = pd.NaT
    for candle in completed:
        if not isinstance(candle, Mapping):
            return False
        timestamp = _coerce_timestamp(candle.get("timestamp"))
        if pd.isna(timestamp):
            return False
        if not pd.isna(previous) and timestamp <= previous:
            return False
        previous = timestamp
    current = _current_minute(engine)
    if pd.isna(current):
        return getattr(engine, "current_candle", None) is None
    return pd.isna(previous) or current > previous


def install_candle_state_hardening(engine_cls: type[Any]) -> None:
    """Install idempotent CandleEngine lifecycle hardening."""
    if bool(getattr(engine_cls, _INSTALLED_ATTR, False)):
        return

    original_init = engine_cls.__init__
    original_replace = engine_cls.replace_history
    original_on_tick = engine_cls.on_tick
    original_finalize = engine_cls._finalize_current_candle
    original_flush = engine_cls.flush
    original_diagnostics = engine_cls.diagnostics

    setattr(engine_cls, _ORIGINAL_INIT_ATTR, original_init)
    setattr(engine_cls, _ORIGINAL_REPLACE_ATTR, original_replace)
    setattr(engine_cls, _ORIGINAL_ON_TICK_ATTR, original_on_tick)
    setattr(engine_cls, _ORIGINAL_FINALIZE_ATTR, original_finalize)
    setattr(engine_cls, _ORIGINAL_FLUSH_ATTR, original_flush)
    setattr(engine_cls, _ORIGINAL_DIAGNOSTICS_ATTR, original_diagnostics)

    def hardened_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _lock_for(self)

    def hardened_replace_history(self: Any, frame: Any) -> None:
        lock = _lock_for(self)
        with lock:
            original_replace(self, frame)
            latest = _latest_completed_minute(self)
            current = _current_minute(self)
            if not pd.isna(latest) and not pd.isna(current) and current <= latest:
                self.current_candle = None
                counters = _counters_for(self)
                counters["history_current_reconcile_total"] += 1
                LOGGER.warning(
                    "CANDLE_CURRENT_RECONCILED_AFTER_HISTORY symbol=%s current_minute=%s latest_completed_minute=%s",
                    getattr(self, "symbol", None) or "symbol_unset",
                    current.isoformat(),
                    latest.isoformat(),
                    extra={
                        "event": "CANDLE_CURRENT_RECONCILED_AFTER_HISTORY",
                        "symbol": getattr(self, "symbol", None) or "symbol_unset",
                        "current_minute": current.isoformat(),
                        "latest_completed_minute": latest.isoformat(),
                        "action": "discarded",
                        "reason": "current_not_newer_than_completed",
                    },
                )
            if not _state_consistent(self):
                from nifty_scalper_bot.data.source import DataIntegrityError

                raise DataIntegrityError("inconsistent candle state after history replacement")

    def hardened_on_tick(self: Any, tick: Mapping[str, Any]) -> Any:
        lock = _lock_for(self)
        with lock:
            try:
                normalized = normalize_market_tick_timestamp(tick)
                tick_timestamp = normalized.timestamp
                tick_minute = tick_timestamp.floor("1min")
                timestamp_source = normalized.source
            except Exception:
                return original_on_tick(self, tick)

            latest = _latest_completed_minute(self)
            if not pd.isna(latest) and tick_minute <= latest:
                counters = _counters_for(self)
                counters["finalized_minute_tick_reject_total"] += 1
                symbol = (
                    tick.get("symbol")
                    or tick.get("trading_symbol")
                    or getattr(self, "symbol", None)
                    or "symbol_unset"
                )
                log_throttled(
                    LOGGER,
                    f"candle_tick_finalized_minute:{symbol}:{tick_minute.isoformat()}",
                    "CANDLE_TICK_REJECTED_FINALIZED_MINUTE symbol=%s tick_minute=%s latest_completed_minute=%s"
                    % (symbol, tick_minute.isoformat(), latest.isoformat()),
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "CANDLE_TICK_REJECTED_FINALIZED_MINUTE",
                        "symbol": symbol,
                        "tick_timestamp": tick_timestamp.isoformat(),
                        "tick_minute": tick_minute.isoformat(),
                        "latest_completed_minute": latest.isoformat(),
                        "timestamp_source": timestamp_source,
                    },
                )
                return None
            return original_on_tick(self, tick)

    def hardened_finalize(self: Any) -> Any:
        lock = _lock_for(self)
        with lock:
            try:
                return original_finalize(self)
            finally:
                # A finalized, rejected, out-of-order, or conflicting candle is
                # never reusable. on_tick immediately installs the next minute
                # after a successful rollover.
                self.current_candle = None

    def hardened_flush(self: Any) -> Any:
        lock = _lock_for(self)
        with lock:
            # Call the original flush; its call to _finalize_current_candle is
            # routed through hardened_finalize under the same reentrant lock.
            return original_flush(self)

    def hardened_diagnostics(self: Any) -> dict[str, Any]:
        lock = _lock_for(self)
        with lock:
            diagnostics = dict(original_diagnostics(self))
            counters = _counters_for(self)
            latest = _latest_completed_minute(self)
            current = _current_minute(self)
            diagnostics.update(
                {
                    "finalized_minute_tick_reject_total": counters[
                        "finalized_minute_tick_reject_total"
                    ],
                    "history_current_reconcile_total": counters[
                        "history_current_reconcile_total"
                    ],
                    "last_completed_timestamp": None
                    if pd.isna(latest)
                    else latest.isoformat(),
                    "current_candle_timestamp": None
                    if pd.isna(current)
                    else current.isoformat(),
                    "state_consistent": _state_consistent(self),
                }
            )
            return diagnostics

    def is_state_consistent(self: Any) -> bool:
        lock = _lock_for(self)
        with lock:
            return _state_consistent(self)

    engine_cls.__init__ = hardened_init
    engine_cls.replace_history = hardened_replace_history
    engine_cls._replace_completed_candles = hardened_replace_history
    engine_cls.on_tick = hardened_on_tick
    engine_cls._finalize_current_candle = hardened_finalize
    engine_cls.flush = hardened_flush
    engine_cls.diagnostics = hardened_diagnostics
    engine_cls.is_state_consistent = is_state_consistent
    setattr(engine_cls, _INSTALLED_ATTR, True)


__all__ = ["install_candle_state_hardening"]
