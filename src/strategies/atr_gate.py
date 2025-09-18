"""Helpers for evaluating ATR percentage guardrails."""

from __future__ import annotations

import logging
from math import isfinite

from src.signals.patches import resolve_atr_band

logger = logging.getLogger(__name__)


def check_atr(
    atr_pct: float,
    cfg: object,
    symbol: str | None,
) -> tuple[bool, str | None, float, float]:
    """Return whether ``atr_pct`` lies within configured ATR guardrails."""

    atr_value = float(atr_pct)

    min_val, max_val = resolve_atr_band(cfg, symbol)

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
