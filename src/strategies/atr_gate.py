"""Helpers for evaluating ATR percentage guardrails."""

from __future__ import annotations

import logging
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATR_PCT: float = 2.0


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` converted to ``float`` when possible."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _iter_sources(cfg: object | Sequence[object] | None) -> Iterable[object]:
    """Yield potential containers holding ATR configuration.

    ``cfg`` may be a single object, mapping, dataclass or a sequence of nested
    objects.  The helper walks through ``raw`` attributes (as used by
    :class:`~src.strategies.strategy_config.StrategyConfig`) and ``model_dump``
    dictionaries from Pydantic models to make the resolution tolerant to the
    different configuration shapes used across the codebase.
    """

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


def _prefer_threshold_min(cfg: object, symbol: str | None) -> float | None:
    """Resolve the minimum ATR%% from threshold configuration."""

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


def _gates_band(cfg: object) -> Tuple[float | None, float | None]:
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


def check_atr(
    atr_pct: float,
    cfg: object,
    symbol: str | None,
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

    Returns
    -------
    tuple[bool, str | None, float, float]
        ``(ok, reason, min_val, max_val)`` where ``reason`` provides context
        when the ATR percentage falls outside the inclusive band.
    """

    atr_value = float(atr_pct)

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

    min_val = float(min_threshold if min_threshold is not None else gates_min or 0.0)

    raw_max = gates_max if gates_max is not None else DEFAULT_MAX_ATR_PCT
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
