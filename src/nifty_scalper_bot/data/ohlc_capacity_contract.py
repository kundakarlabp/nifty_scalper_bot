"""Decouple completed OHLC retention from raw tick retention.

The MarketDataManager intentionally keeps only a small raw-tick cache to bound
memory on the production Lightsail host. ORB, however, needs the 09:15 opening
range to remain available for the full NSE session. This adapter enlarges only
completed one-minute candle retention; raw tick retention is untouched.
"""

from __future__ import annotations

from collections import defaultdict, deque
from functools import wraps
import os
from typing import Any, Mapping

_DEFAULT_OHLC_CAPACITY = 500
_MIN_SESSION_CAPACITY = 400


def configured_ohlc_capacity() -> int:
    """Return bounded completed-candle capacity required by intraday strategies."""
    try:
        configured = int(
            float(os.getenv("MDM_OHLC_CACHE_LEN", str(_DEFAULT_OHLC_CAPACITY)))
        )
    except (TypeError, ValueError):
        configured = _DEFAULT_OHLC_CAPACITY
    return max(_MIN_SESSION_CAPACITY, configured)


def _resize(value: Any, capacity: int) -> deque[Any]:
    rows = list(value or [])
    return deque(rows[-capacity:], maxlen=capacity)


def ensure_ohlc_capacity(manager: Any) -> int:
    """Enlarge completed-candle storage on an existing MDM instance once."""
    capacity = configured_ohlc_capacity()
    if int(getattr(manager, "_ohlc_capacity_contract_ready", 0) or 0) == capacity:
        return capacity

    setattr(manager, "_ohlc_cache_len", capacity)
    projection = getattr(manager, "_ohlc", None)
    if isinstance(projection, Mapping):
        replacement = defaultdict(lambda: deque(maxlen=capacity))
        for symbol, rows in list(projection.items()):
            replacement[symbol] = _resize(rows, capacity)
        try:
            manager._ohlc = replacement
        except Exception:
            pass

    engines = getattr(manager, "_engines", None)
    if isinstance(engines, Mapping):
        for engine in list(engines.values()):
            completed = getattr(engine, "_completed_candles", None)
            if completed is not None and getattr(completed, "maxlen", None) != capacity:
                try:
                    engine._completed_candles = _resize(completed, capacity)
                except Exception:
                    continue
            try:
                if int(getattr(engine, "max_bars", 0) or 0) < capacity:
                    engine.max_bars = capacity
            except Exception:
                pass

    setattr(manager, "_ohlc_capacity_contract_ready", capacity)
    return capacity


def install_mdm_ohlc_capacity_contract() -> bool:
    """Install the completed-OHLC capacity adapter exactly once."""
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    marker = "_completed_ohlc_capacity_contract_installed"
    if bool(getattr(MarketDataManager, marker, False)):
        return True

    original_init = MarketDataManager.__init__
    original_get_engine = MarketDataManager._get_engine
    original_get_ohlc = MarketDataManager.get_ohlc_bars

    @wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        ensure_ohlc_capacity(self)

    @wraps(original_get_engine)
    def _get_engine(self: Any, symbol: str) -> Any:
        engine = original_get_engine(self, symbol)
        capacity = ensure_ohlc_capacity(self)
        completed = getattr(engine, "_completed_candles", None)
        if completed is not None and getattr(completed, "maxlen", None) != capacity:
            try:
                engine._completed_candles = _resize(completed, capacity)
            except Exception:
                pass
        try:
            if int(getattr(engine, "max_bars", 0) or 0) < capacity:
                engine.max_bars = capacity
        except Exception:
            pass
        return engine

    @wraps(original_get_ohlc)
    def get_ohlc_bars(
        self: Any, symbol: str, *, limit: int | None = None
    ) -> list[Any]:
        """Return canonical completed bars without raw-tick 250-row truncation."""
        capacity = ensure_ohlc_capacity(self)
        requested = capacity if limit is None else max(1, min(int(limit), capacity))

        # Preserve native symbol normalization and projection diagnostics first.
        native = list(original_get_ohlc(self, symbol, limit=requested) or [])
        try:
            engine = self._get_engine(symbol)
            canonical = list(engine.get_completed_bars() or [])
        except Exception:
            canonical = []

        # CandleEngine is MDM's authoritative finalized-history owner. Prefer it
        # only when it proves a deeper canonical view than the legacy projection.
        rows = canonical if len(canonical) > len(native) else native
        if limit is not None:
            rows = rows[-requested:]
        elif len(rows) > capacity:
            rows = rows[-capacity:]

        key_resolver = getattr(self, "_bar_symbol_key", None)
        if callable(key_resolver):
            try:
                key = key_resolver(symbol)
                current = getattr(self, "_ohlc", {}).get(key)
                if (
                    current is None
                    or getattr(current, "maxlen", None) != capacity
                    or len(current) != len(rows)
                ):
                    self._ohlc[key] = _resize(rows, capacity)
            except Exception:
                pass
        return [dict(row) if isinstance(row, Mapping) else row for row in rows]

    MarketDataManager.__init__ = __init__  # type: ignore[method-assign]
    MarketDataManager._get_engine = _get_engine  # type: ignore[method-assign]
    MarketDataManager.get_ohlc_bars = get_ohlc_bars  # type: ignore[method-assign]
    setattr(MarketDataManager, marker, True)
    return True


__all__ = [
    "configured_ohlc_capacity",
    "ensure_ohlc_capacity",
    "install_mdm_ohlc_capacity_contract",
]
