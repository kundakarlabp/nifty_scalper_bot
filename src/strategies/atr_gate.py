"""Helpers for evaluating ATR percentage guardrails."""

from __future__ import annotations

import logging
from math import isfinite
from src.signals.patches import resolve_atr_band

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATR_PCT: float = 2.0
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

    logger.info(
        "ATR gate: atr_pct=%.4f min=%.4f max=%s result=%s",
        atr_value,
        min_val,
        max_repr,
        "ok" if ok else "out_of_band",
    )

    return ok, reason, min_val, max_val


__all__ = ["check_atr"]
