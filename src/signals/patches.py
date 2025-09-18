"""Signal-level helper utilities and runtime patches."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence, Tuple

DEFAULT_MAX_ATR_PCT: float = 2.0

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


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` converted to ``float`` when possible."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _iter_sources(cfg: object | Sequence[object] | None) -> Iterable[object]:
    """Yield potential containers holding ATR configuration."""

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
    """Return ``key`` from ``container`` handling mappings and attribute access."""

    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def resolve_atr_band(cfg: object, symbol: str | None) -> Tuple[float, float]:
    """Return the configured ATR%% band for ``symbol``.

    Parameters
    ----------
    cfg:
        Strategy configuration object (dataclass, namespace or mapping).
    symbol:
        Underlying symbol used to pick the appropriate minimum threshold.

    Returns
    -------
    Tuple[float, float]
        A ``(min_pct, max_pct)`` pair where ``max_pct`` reflects the configured
        upper bound (defaults to :data:`DEFAULT_MAX_ATR_PCT` when unspecified).
    """

    resolved_symbol = (symbol or "").upper()
    primary = (
        "min_atr_pct_banknifty" if "BANK" in resolved_symbol else "min_atr_pct_nifty"
    )
    fallbacks = (primary, "min_atr_pct_nifty", "min_atr_pct_banknifty")

    min_threshold: float | None = None
    for source in _iter_sources(cfg):
        thresholds = _get_value(source, "thresholds")
        if thresholds is None:
            continue
        for key in fallbacks:
            val = _coerce_float(_get_value(thresholds, key))
            if val is not None:
                min_threshold = val
                break
        if min_threshold is not None:
            break

    gates_min: float | None = None
    gates_max: float | None = None
    for source in _iter_sources(cfg):
        gates = _get_value(source, "gates")
        if gates is None:
            continue
        if gates_min is None:
            gates_min = _coerce_float(_get_value(gates, "atr_pct_min"))
        if gates_max is None:
            gates_max = _coerce_float(_get_value(gates, "atr_pct_max"))
        if gates_min is not None and gates_max is not None:
            break

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

    min_val = float(min_threshold if min_threshold is not None else gates_min or 0.0)
    raw_max = gates_max if gates_max is not None else DEFAULT_MAX_ATR_PCT
    if raw_max is None:
        raw_max = DEFAULT_MAX_ATR_PCT
    max_val = float(raw_max)

    return min_val, max_val


__all__ = ["check_atr_band", "resolve_atr_band", "DEFAULT_MAX_ATR_PCT"]
