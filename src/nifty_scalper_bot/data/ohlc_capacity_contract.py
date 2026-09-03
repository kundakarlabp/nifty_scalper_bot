"""Decouple completed OHLC projection retention from raw tick retention.

MarketDataManager intentionally keeps a small raw-tick cache to bound memory on
the production host. CandleEngine already owns up to 500 finalized one-minute
bars, but MDM's read-only projection was truncated to the raw-tick cache length.
ORB therefore lost the 09:15 opening range late in the session.

This adapter changes only the normal MDM projection refresh path. CandleEngine
remains the sole owner of finalized history and MDM reads never regenerate a
projection that has been deliberately removed for diagnostics/tests.
"""

from __future__ import annotations

from collections import deque
from functools import wraps
import os
from typing import Any, Mapping

_DEFAULT_OHLC_CAPACITY = 500
_MIN_SESSION_CAPACITY = 400
_MAX_NATIVE_CAPACITY = 500


def configured_ohlc_capacity() -> int:
    """Return completed-projection capacity supported by native CandleEngine."""
    try:
        configured = int(
            float(os.getenv("MDM_OHLC_CACHE_LEN", str(_DEFAULT_OHLC_CAPACITY)))
        )
    except (TypeError, ValueError):
        configured = _DEFAULT_OHLC_CAPACITY
    return min(max(_MIN_SESSION_CAPACITY, configured), _MAX_NATIVE_CAPACITY)


def _expanded_projection(
    manager: Any,
    symbol: str,
    *,
    source: str | None,
    native_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project deeper canonical history after the native refresh write path."""
    capacity = configured_ohlc_capacity()
    try:
        engine = manager.get_candle_engine(symbol)
        completed = list(engine.get_completed_bars() or [])
    except Exception:
        return native_rows

    if len(completed) <= len(native_rows):
        return native_rows

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

    original_refresh = MarketDataManager._refresh_candle_projection

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

    MarketDataManager._refresh_candle_projection = _refresh_candle_projection  # type: ignore[method-assign]
    setattr(MarketDataManager, marker, True)
    return True


__all__ = [
    "configured_ohlc_capacity",
    "install_mdm_ohlc_capacity_contract",
]
