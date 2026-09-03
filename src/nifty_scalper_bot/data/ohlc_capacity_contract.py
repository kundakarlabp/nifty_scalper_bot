"""Decouple completed OHLC projection retention from raw tick retention.

MarketDataManager intentionally keeps a small raw-tick cache to bound memory on
the production host. CandleEngine already owns up to 500 finalized one-minute
bars, but MDM's read-only projection was truncated to the raw-tick cache length.
ORB therefore lost the 09:15 opening range late in the session.

This adapter changes only the normal MDM projection refresh path. CandleEngine
remains the sole owner of finalized history and MDM reads never regenerate a
projection that has been deliberately removed for diagnostics/tests. Explicit
legacy ``cache_len`` overrides remain authoritative for callers that deliberately
use the historical combined tick/OHLC capacity contract.
"""

from __future__ import annotations

from collections import deque
from functools import wraps
import os
from typing import Any, Mapping

_DEFAULT_OHLC_CAPACITY = 500
_MIN_SESSION_CAPACITY = 400
_MAX_NATIVE_CAPACITY = 500
_LEGACY_DEFAULT_CACHE_LEN = 1000


def configured_ohlc_capacity() -> int:
    """Return completed-projection capacity supported by native CandleEngine."""
    try:
        configured = int(
            float(os.getenv("MDM_OHLC_CACHE_LEN", str(_DEFAULT_OHLC_CAPACITY)))
        )
    except (TypeError, ValueError):
        configured = _DEFAULT_OHLC_CAPACITY
    return min(max(_MIN_SESSION_CAPACITY, configured), _MAX_NATIVE_CAPACITY)


def _projection_capacity(manager: Any) -> int:
    """Resolve decoupled capacity while preserving explicit legacy overrides."""
    configured = configured_ohlc_capacity()
    try:
        current_raw = max(1, int(getattr(manager, "_cache_len", configured) or configured))
    except (TypeError, ValueError):
        current_raw = configured

    baseline = getattr(manager, "_ohlc_raw_cache_baseline", None)
    legacy_explicit = bool(
        getattr(manager, "_ohlc_legacy_explicit_cache_len", False)
    )
    try:
        baseline_value = int(baseline) if baseline is not None else None
    except (TypeError, ValueError):
        baseline_value = None

    # ``MDM_TICK_CACHE_LEN`` is intentionally decoupled from completed OHLC.
    # An explicit constructor ``cache_len=...`` or a later direct runtime
    # override retains the historical combined-capacity semantics used by
    # diagnostics/tests and specialised callers.
    if legacy_explicit or (
        baseline_value is not None and current_raw != baseline_value
    ):
        return min(configured, current_raw)
    return configured


def _expanded_projection(
    manager: Any,
    symbol: str,
    *,
    source: str | None,
    native_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project deeper canonical history after the native refresh write path."""
    capacity = _projection_capacity(manager)
    try:
        engine = manager.get_candle_engine(symbol)
        completed = list(engine.get_completed_bars() or [])
    except Exception:
        return native_rows

    # Never widen an explicit legacy projection cap. The native refresh may
    # already satisfy it; otherwise project the same canonical suffix at the
    # resolved capacity.
    if len(native_rows) >= min(len(completed), capacity):
        return native_rows[-capacity:]

    normalized = (
        manager._canonical_symbol(symbol)
        if callable(getattr(manager, "_canonical_symbol", None))
        else str(symbol)
    )
    key = (
        manager._bar_symbol_key(normalized)
        if callable(getattr(manager, "_bar_symbol_key", None))
        else normalized
    )
    projected: deque[dict[str, Any]] = deque(maxlen=capacity)
    for row in completed[-capacity:]:
        bar = dict(row)
        bar["symbol"] = normalized
        bar["source"] = source or bar.get("source") or "candle_engine"
        projected.append(bar)

    with manager._lock:
        manager._ohlc[key] = projected
        metrics = getattr(manager, "_candle_metrics", None)
        if isinstance(metrics, Mapping):
            try:
                metrics["candle_projection_size"] = float(len(projected))
            except Exception:
                pass
        diagnostics = getattr(manager, "_candle_projection_diagnostics", None)
        if isinstance(diagnostics, dict):
            current = diagnostics.get(normalized)
            if isinstance(current, dict):
                current["projection_size"] = len(projected)

    return [dict(row) for row in projected]


def install_mdm_ohlc_capacity_contract() -> bool:
    """Install the completed-OHLC projection adapter exactly once."""
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    marker = "_completed_ohlc_capacity_contract_installed"
    if bool(getattr(MarketDataManager, marker, False)):
        return True

    original_init = MarketDataManager.__init__
    original_refresh = MarketDataManager._refresh_candle_projection

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        requested_cache_len = kwargs.get("cache_len", _LEGACY_DEFAULT_CACHE_LEN)
        original_init(self, *args, **kwargs)
        try:
            baseline = max(1, int(getattr(self, "_cache_len", 1) or 1))
        except (TypeError, ValueError):
            baseline = 1
        self._ohlc_raw_cache_baseline = baseline
        try:
            explicit_value = int(requested_cache_len)
        except (TypeError, ValueError):
            explicit_value = _LEGACY_DEFAULT_CACHE_LEN
        self._ohlc_legacy_explicit_cache_len = (
            explicit_value != _LEGACY_DEFAULT_CACHE_LEN
        )

    @wraps(original_refresh)
    def _refresh_candle_projection(
        self: Any, symbol: str, *, source: str | None = None
    ) -> list[dict[str, Any]]:
        native_rows = list(original_refresh(self, symbol, source=source) or [])
        return _expanded_projection(
            self,
            symbol,
            source=source,
            native_rows=native_rows,
        )

    MarketDataManager.__init__ = __init__  # type: ignore[method-assign]
    MarketDataManager._refresh_candle_projection = _refresh_candle_projection  # type: ignore[method-assign]
    setattr(MarketDataManager, marker, True)
    return True


__all__ = [
    "configured_ohlc_capacity",
    "install_mdm_ohlc_capacity_contract",
]
