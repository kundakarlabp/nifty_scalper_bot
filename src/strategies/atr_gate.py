"""Helpers for evaluating ATR percentage guardrails."""

from __future__ import annotations

import logging
import os
import time
from math import isfinite
from threading import Lock

from src.signals.patches import resolve_atr_band

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATR_PCT: float = 2.0
try:
    # Default to logging no more than once every 30 seconds to reduce noise.
    MIN_INTERVAL_S: float = float(os.getenv("ATR_LOG_INTERVAL_S", "30"))
except ValueError:
    MIN_INTERVAL_S = 30.0

_THROTTLE_LOCK: Lock = Lock()
_LAST_LOG_STATE: dict[str, tuple[bool, float, float | None, float]] = {}
_LAST_ATR_LOG_TS: float = 0.0


def _should_log_atr(now: float, min_interval_s: float) -> bool:
    """Return ``True`` when the ATR info log should fire based on time."""

    global _LAST_ATR_LOG_TS
    if now - _LAST_ATR_LOG_TS >= min_interval_s:
        _LAST_ATR_LOG_TS = now
        return True
    return False


def _normalise_log_state(
    symbol: str | None,
    *,
    atr_value: float,
    min_val: float,
    max_bound: float,
    ok: bool,
) -> tuple[str, tuple[bool, float, float | None, float]]:
    """Return a hashable representation of the current log state.

    Repeated calls with the same inputs produce identical keys which enables
    throttling for the very chatty ATR gate logger. ATR values are rounded to
    four decimal places to avoid noisy floating point jitter while still
    surfacing meaningful changes.
    """

    key = symbol or "__default__"
    max_marker: float | None
    if isfinite(max_bound):
        max_marker = round(max_bound, 4)
    else:
        max_marker = None
    state = (
        ok,
        round(min_val, 4),
        max_marker,
        round(atr_value, 4),
    )
    return key, state


def _should_log_info(
    symbol: str | None,
    *,
    atr_value: float,
    min_val: float,
    max_bound: float,
    ok: bool,
) -> bool:
    """Return ``True`` when the throttled info log should fire."""

    global _LAST_ATR_LOG_TS
    key, state = _normalise_log_state(
        symbol, atr_value=atr_value, min_val=min_val, max_bound=max_bound, ok=ok
    )
    with _THROTTLE_LOCK:
        previous = _LAST_LOG_STATE.get(key)
        now = time.time()
        if previous != state:
            _LAST_LOG_STATE[key] = state
            _LAST_ATR_LOG_TS = now
            return True
        if _should_log_atr(now, MIN_INTERVAL_S):
            return True
    return False


def _reset_log_throttle_state() -> None:
    """Clear the memoised log state.

    This is primarily intended for test isolation.
    """

    global _LAST_ATR_LOG_TS
    with _THROTTLE_LOCK:
        _LAST_LOG_STATE.clear()
        _LAST_ATR_LOG_TS = 0.0


def check_atr(
    atr_pct: float,
    cfg: object,
    symbol: str | None,
    band: tuple[float | None, float | None] | None = None,
) -> tuple[bool, str | None, float, float]:
    """Return whether ``atr_pct`` lies within configured ATR guardrails.

    Parameters
    ----------
    atr_pct:
        Observed ATR percentage for the underlying.
    cfg:
        Strategy configuration object (dataclass, namespace or mapping).
    symbol:
        Underlying symbol used to pick the appropriate minimum threshold.
    band:
        Optional precomputed ``(min, max)`` ATR band to evaluate against.

    Returns
    -------
    tuple[bool, str | None, float, float]
        ``(ok, reason, min_val, max_val)`` where ``reason`` provides context
        when the ATR percentage falls outside the inclusive band.
    """

    atr_value = float(atr_pct)

    min_candidate: float | None
    max_candidate: float | None
    if band is not None:
        try:
            min_candidate = float(band[0]) if band[0] is not None else None
        except (IndexError, TypeError, ValueError):
            min_candidate = None
        try:
            max_candidate = float(band[1]) if band[1] is not None else None
        except (IndexError, TypeError, ValueError):
            max_candidate = None
        if min_candidate is None or max_candidate is None:
            min_candidate, max_candidate = resolve_atr_band(
                cfg, symbol, default_max=DEFAULT_MAX_ATR_PCT
            )
    else:
        min_candidate, max_candidate = resolve_atr_band(
            cfg, symbol, default_max=DEFAULT_MAX_ATR_PCT
        )

    min_val = float(min_candidate if min_candidate is not None else 0.0)

    raw_max = max_candidate if max_candidate is not None else DEFAULT_MAX_ATR_PCT
    if raw_max is None:
        raw_max = DEFAULT_MAX_ATR_PCT
    max_val = float(raw_max)

    effective_max = float("inf") if max_val <= 0 else max(max_val, min_val)

    ok = min_val <= atr_value <= effective_max
    reason: str | None = None
    if not ok:
        if atr_value < min_val:
            reason = f"atr_out_of_band: atr={atr_value:.4f} < min={min_val}"
        else:
            bound = effective_max if isfinite(effective_max) else max_val
            reason = f"atr_out_of_band: atr={atr_value:.4f} > max={bound}"

    max_repr: str
    if isfinite(effective_max):
        max_repr = f"{effective_max:.4f}"
    else:
        max_repr = "inf"

    info_log = _should_log_info(
        symbol,
        atr_value=atr_value,
        min_val=min_val,
        max_bound=effective_max,
        ok=ok,
    )

    log_fn = logger.info if info_log or not ok else logger.debug
    log_fn(
        "ATR gate: atr_pct=%.4f min=%.4f max=%s result=%s",
        atr_value,
        min_val,
        max_repr,
        "ok" if ok else "out_of_band",
    )

    return ok, reason, min_val, max_val


__all__ = ["check_atr"]
