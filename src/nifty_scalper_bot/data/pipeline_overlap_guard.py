"""Guard the hydration/live candle seam from noisy false out-of-order alerts.

Historical reseed can legitimately have the same or slightly newer last minute
than the first live bar finalized by the builder.  The old CandleStore.push()
treated every older candle as a warning-grade integrity violation.  This module
keeps strict monotonic enforcement for genuinely stale candles, while quietly
dropping a small overlap window at the hydration/live boundary.
"""

from __future__ import annotations

from collections import deque
import os
from typing import Any

import pandas as pd


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def install_candle_store_overlap_guard() -> bool:
    """Patch CandleStore.push once with a bounded overlap tolerance."""
    try:
        from nifty_scalper_bot.data import pipeline as pipeline_module
    except Exception:
        return False

    CandleStore = getattr(pipeline_module, "CandleStore", None)
    if CandleStore is None:
        return False
    current = getattr(CandleStore, "push", None)
    if current is None or getattr(current, "_overlap_guard_installed", False):
        return False

    def push(self: Any, candle: Any) -> None:
        with self._lock:
            if candle.symbol not in self._store:
                self._store[candle.symbol] = deque(maxlen=self._maxlen)
            buf = self._store[candle.symbol]
            if buf:
                last_ts = pd.Timestamp(buf[-1].timestamp).floor("1min")
                incoming_ts = pd.Timestamp(candle.timestamp).floor("1min")
                if incoming_ts == last_ts:
                    return
                if incoming_ts < last_ts:
                    overlap_seconds = max(
                        0.0,
                        _float_env("PIPELINE_CANDLE_OVERLAP_TOLERANCE_SECONDS", 120.0),
                    )
                    age_delta = max(0.0, (last_ts - incoming_ts).total_seconds())
                    if age_delta <= overlap_seconds:
                        pipeline_module.LOGGER.debug(
                            "candle_store_overlap_duplicate symbol=%s incoming_ts=%s last_ts=%s age_delta_s=%.1f",
                            candle.symbol,
                            incoming_ts.isoformat(),
                            last_ts.isoformat(),
                            age_delta,
                            extra={
                                "event": "candle_store_overlap_duplicate",
                                "symbol": candle.symbol,
                                "incoming_ts": incoming_ts.isoformat(),
                                "last_ts": last_ts.isoformat(),
                                "age_delta_s": age_delta,
                                "source": "candle_store",
                                "reason": "hydration_live_boundary_overlap",
                            },
                        )
                        return

                    pipeline_module._DROPPED_CANDLES.increment()
                    pipeline_module.log_throttled(
                        pipeline_module.LOGGER,
                        f"candle_store_out_of_order:{candle.symbol}",
                        (
                            "candle_store_out_of_order symbol=%s incoming_ts=%s "
                            "last_ts=%s source=candle_store"
                        )
                        % (candle.symbol, incoming_ts.isoformat(), last_ts.isoformat()),
                        interval_sec=30.0,
                        level=pipeline_module.logging.WARNING,
                        extra={
                            "event": "candle_store_out_of_order",
                            "symbol": candle.symbol,
                            "incoming_ts": incoming_ts.isoformat(),
                            "last_ts": last_ts.isoformat(),
                            "source": "candle_store",
                            "total_dropped": pipeline_module._DROPPED_CANDLES.value,
                        },
                    )
                    raise pipeline_module.DataIntegrityError("candle store timestamps must be monotonic")
            buf.append(candle)

    push.__name__ = getattr(current, "__name__", "push")
    push.__doc__ = getattr(current, "__doc__", None)
    setattr(push, "_overlap_guard_installed", True)
    setattr(push, "_original", current)
    CandleStore.push = push  # type: ignore[assignment]
    return True
