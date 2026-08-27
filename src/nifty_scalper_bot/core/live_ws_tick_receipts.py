"""Preserve genuine current-generation WebSocket receipt proof for LIVE execution.

PR #1157 intentionally fans current-generation WS ticks to DataHub/Runner even
when the MDM cache rejects an older event timestamp. Those ticks are valid live
market-data receipts, but a rejected cache write does not append them to MDM's
raw tick history. Execution candidate readiness therefore saw
``real_ticks_last_60s == 0`` despite an active WS stream.

This adapter records only WS receipts that MDM itself accepted as belonging to
the current subscription generation. It does not alter quote/cache ordering,
strategy thresholds, risk gates, or order routing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from functools import wraps
import time
from typing import Any, Mapping

_PATCH_ATTR = "_live_ws_tick_receipt_patch_installed"
_RECEIPTS_ATTR = "_live_ws_receipts_60s"
_WS_SOURCES = frozenset({"ws", "ws_full", "websocket", "stream"})
_WINDOW_SECONDS = 60.0
_MAX_RECEIPTS_PER_SYMBOL = 4096


def _prune(receipts: deque[float], now_mono: float) -> None:
    cutoff = now_mono - _WINDOW_SECONDS
    while receipts and receipts[0] < cutoff:
        receipts.popleft()


def apply_patch() -> bool:
    """Install the MDM live-WS receipt counter idempotently."""
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    if bool(getattr(MarketDataManager, _PATCH_ATTR, False)):
        return True

    original_emit = MarketDataManager._emit_tick
    original_snapshot = MarketDataManager.get_symbol_snapshot

    @wraps(original_emit)
    def _emit_tick(
        self: Any,
        symbol: str,
        tick: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            canonical = self._canonical_symbol(symbol)
        except Exception:
            canonical = str(symbol or "")
        source = str(
            kwargs.get("source")
            or (tick.get("source") if isinstance(tick, Mapping) else None)
            or "ws"
        ).strip().lower()
        before = float(
            (getattr(self, "_last_valid_live_tick_mono", {}) or {}).get(canonical, 0.0)
            or 0.0
        )
        call_started = time.monotonic()
        result = original_emit(self, symbol, tick, *args, **kwargs)

        if source not in _WS_SOURCES:
            return result
        after = float(
            (getattr(self, "_last_valid_live_tick_mono", {}) or {}).get(canonical, 0.0)
            or 0.0
        )
        # MDM advances this marker only after accepting a WS tick for the current
        # subscription generation. A cache reject after that point is still valid
        # LIVE receipt proof; stale-generation ticks do not advance the marker.
        if after <= before or after < call_started:
            return result

        now_mono = time.monotonic()
        lock = getattr(self, "_lock", None)
        if lock is None:
            return result
        with lock:
            by_symbol = getattr(self, _RECEIPTS_ATTR, None)
            if by_symbol is None:
                by_symbol = {}
                setattr(self, _RECEIPTS_ATTR, by_symbol)
            receipts = by_symbol.get(canonical)
            if receipts is None:
                receipts = deque(maxlen=_MAX_RECEIPTS_PER_SYMBOL)
                by_symbol[canonical] = receipts
            receipts.append(now_mono)
            _prune(receipts, now_mono)
        return result

    @wraps(original_snapshot)
    def get_symbol_snapshot(self: Any, symbol: str) -> Any:
        snapshot = original_snapshot(self, symbol)
        try:
            canonical = self._canonical_symbol(symbol)
        except Exception:
            canonical = str(symbol or "")
        now_mono = time.monotonic()
        receipt_count = 0
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                by_symbol = getattr(self, _RECEIPTS_ATTR, None) or {}
                receipts = by_symbol.get(canonical)
                if receipts is not None:
                    _prune(receipts, now_mono)
                    receipt_count = len(receipts)
        try:
            existing = max(0, int(getattr(snapshot, "real_ticks_last_60s", 0) or 0))
        except (TypeError, ValueError):
            existing = 0
        if receipt_count <= existing:
            return snapshot
        return replace(snapshot, real_ticks_last_60s=receipt_count)

    MarketDataManager._emit_tick = _emit_tick  # type: ignore[method-assign]
    MarketDataManager.get_symbol_snapshot = get_symbol_snapshot  # type: ignore[method-assign]
    setattr(MarketDataManager, _PATCH_ATTR, True)
    return True


__all__ = ["apply_patch"]
