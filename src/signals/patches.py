"""Signal-level helper utilities and runtime patches."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Tuple

_BAND_EPSILON = 1e-4


def _within_band_floor(value: float, bound: float) -> bool:
    """Return ``True`` when ``value`` respects the lower ``bound`` within tolerance."""

    return value >= bound or math.isclose(value, bound, abs_tol=_BAND_EPSILON)


def _within_band_ceiling(value: float, bound: float) -> bool:
    """Return ``True`` when ``value`` respects the upper ``bound`` within tolerance."""

    return value <= bound or math.isclose(value, bound, abs_tol=_BAND_EPSILON)


def check_atr_band(atr_pct: float, min_val: float, max_val: float) -> tuple[bool, str | None]:
    """Return whether ``atr_pct`` falls within the inclusive ``[min_val, max_val]`` band.

    Parameters
    ----------
    atr_pct:
        Observed ATR percentage of the underlying instrument.
    min_val:
        Lower bound for the acceptable ATR percentage.
    max_val:
        Upper bound for the acceptable ATR percentage.

    Returns
    -------
    Tuple[bool, str | None]
        ``(True, None)`` when within the configured band, otherwise ``(False, reason)``
        where ``reason`` contains a diagnostic message including the offending value
        and configured limits.
    """

    min_bound = float(min_val)
    max_bound = float(max_val)
    if max_bound <= 0:
        max_bound = float("inf")
    elif max_bound < min_bound:
        max_bound = min_bound

    if not _within_band_floor(atr_pct, min_bound):
        return False, f"atr_out_of_band: atr={atr_pct:.4f} < min={min_bound}"
    if max_bound != float("inf") and not _within_band_ceiling(atr_pct, max_bound):
        return False, f"atr_out_of_band: atr={atr_pct:.4f} > max={max_bound}"
    return True, None


def _lookup(source: Any, key: str) -> Any:
    """Best-effort attribute/key lookup that preserves ``None`` when missing."""

    if source is None:
        return None
    if isinstance(source, Mapping):
        if key in source:
            return source[key]
        return None
    if hasattr(source, key):
        return getattr(source, key)
    if hasattr(source, "__dict__"):
        data = vars(source)
        if key in data:
            return data[key]
    return None


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def _first_present_float(source: Any, keys: Iterable[str]) -> float | None:
    """Return the first key present in ``source`` coerced to ``float``."""

    for key in keys:
        val = _lookup(source, key)
        if val is not None:
            coerced = _coerce_float(val)
            if coerced is not None:
                return coerced
    return None


def resolve_atr_band(
    cfg: Any | None,
    *,
    symbol: str | None = None,
    gates: Any | None = None,
) -> Tuple[float, float]:
    """Return the effective ``(min_pct, max_pct)`` ATR guard rails.

    A single resolver keeps ATR percentages consistent everywhere by preferring
    instrument-aware thresholds and falling back to legacy gate controls only
    when those tuned minima are absent.  The hierarchy is therefore:

    1. ``cfg.raw["thresholds"]`` specific to the instrument symbol.
    2. ``gates`` provided either at runtime or inside ``cfg.raw``.
    3. Dataclass attributes (``cfg.min_atr_pct_*`` / ``cfg.atr_min``) for
       backwards compatibility with tests or deserialised objects.
    """

    raw_cfg = _lookup(cfg, "raw")
    raw_thresholds = raw_cfg.get("thresholds", {}) if isinstance(raw_cfg, Mapping) else {}
    raw_gates = raw_cfg.get("gates", {}) if isinstance(raw_cfg, Mapping) else {}
    merged_gates = gates if gates is not None else raw_gates

    symbol_str = str(symbol or "").upper()
    primary_key = "min_atr_pct_banknifty" if "BANK" in symbol_str else "min_atr_pct_nifty"
    min_key_order = [primary_key, "min_atr_pct_nifty", "min_atr_pct_banknifty", "min_atr_pct"]

    min_pct = _first_present_float(raw_thresholds, min_key_order)
    if min_pct is None:
        min_pct = _first_present_float(merged_gates, ["atr_pct_min", "atr_min", "min_atr_pct"])
    if min_pct is None:
        min_pct = _first_present_float(cfg, min_key_order)
    if min_pct is None:
        min_pct = _first_present_float(cfg, ["atr_min", "atr_pct_min"]) or 0.0

    max_pct = _first_present_float(merged_gates, ["atr_pct_max", "atr_max"])
    if max_pct is None:
        max_pct = _first_present_float(cfg, ["atr_pct_max", "atr_max"]) or 0.0

    min_val = float(min_pct)
    max_val = float(max_pct)
    if 0 < max_val < min_val:
        max_val = min_val
    return min_val, max_val


__all__ = ["check_atr_band", "resolve_atr_band"]
