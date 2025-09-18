"""Signal-level helper utilities and runtime patches."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_BAND_EPSILON = 1e-4


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` converted to ``float`` when possible."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _iter_sources(cfg: object | Sequence[object] | None) -> Iterable[object]:
    """Yield candidate containers that may hold ATR configuration."""

    if cfg is None:
        return

    if isinstance(cfg, Sequence) and not isinstance(cfg, (str, bytes)):  # type: ignore[misc]
        for item in cfg:
            yield from _iter_sources(item)
        return

    yield cfg

    raw = getattr(cfg, "raw", None)
    if raw is not None:
        yield raw

    if hasattr(cfg, "model_dump"):
        try:
            dumped = cfg.model_dump()
        except Exception:  # pragma: no cover - defensive fallback
            dumped = None
        if dumped:
            yield dumped


def _get_value(container: object, key: str) -> Any:
    """Return ``key`` from ``container`` supporting mappings and attributes."""

    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _prefer_threshold_min(cfg: object, symbol: str | None) -> float | None:
    """Resolve the minimum ATR percentage from threshold configuration."""

    symbol = (symbol or "").upper()
    primary = "min_atr_pct_banknifty" if "BANK" in symbol else "min_atr_pct_nifty"
    fallbacks = (primary, "min_atr_pct_nifty", "min_atr_pct_banknifty")

    for source in _iter_sources(cfg):
        thresholds = _get_value(source, "thresholds")
        if thresholds is None:
            continue
        for key in fallbacks:
            val = _coerce_float(_get_value(thresholds, key))
            if val is not None:
                return val
    return None


def _gates_band(cfg: object) -> tuple[float | None, float | None]:
    """Return ``(min, max)`` values from gate configuration when available."""

    min_val: float | None = None
    max_val: float | None = None

    for source in _iter_sources(cfg):
        gates = _get_value(source, "gates")
        if gates is None:
            continue
        if min_val is None:
            min_val = _coerce_float(_get_value(gates, "atr_pct_min"))
        if max_val is None:
            max_val = _coerce_float(_get_value(gates, "atr_pct_max"))
        if min_val is not None and max_val is not None:
            break
    return min_val, max_val


def resolve_atr_band(
    cfg: object,
    symbol: str | None,
    *,
    default_min: float = 0.0,
    default_max: float = 2.0,
) -> tuple[float, float]:
    """Return the configured ``(min_pct, max_pct)`` ATR band."""

    min_threshold = _prefer_threshold_min(cfg, symbol)
    gates_min, gates_max = _gates_band(cfg)

    if gates_min is None:
        for source in _iter_sources(cfg):
            candidate = _coerce_float(_get_value(source, "atr_pct_min"))
            if candidate is not None:
                gates_min = candidate
                break
    if gates_min is None:
        for source in _iter_sources(cfg):
            candidate = _coerce_float(_get_value(source, "atr_min"))
            if candidate is not None:
                gates_min = candidate
                break

    if gates_max is None:
        for source in _iter_sources(cfg):
            candidate = _coerce_float(_get_value(source, "atr_pct_max"))
            if candidate is not None:
                gates_max = candidate
                break
    if gates_max is None:
        for source in _iter_sources(cfg):
            candidate = _coerce_float(_get_value(source, "atr_max"))
            if candidate is not None:
                gates_max = candidate
                break

    min_val = float(min_threshold if min_threshold is not None else gates_min or default_min)

    raw_max = gates_max if gates_max is not None else default_max
    if raw_max is None:
        raw_max = default_max
    max_val = float(raw_max)

    return min_val, max_val


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


__all__ = ["check_atr_band", "resolve_atr_band"]
