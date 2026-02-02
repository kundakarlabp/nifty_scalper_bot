# src/nifty_scalper_bot/indicators/atr_provider.py

from dataclasses import dataclass
from time import time
import threading
import math
from nifty_scalper_bot.utils.logging import get_logger

@dataclass
class ATRSnapshot:
    """Validated ATR data with metadata"""
    value: float
    timestamp: float
    period: int = 14
    source: str = "indicator_engine"
    
    @property
    def age_seconds(self) -> float:
        return time() - self.timestamp
    
    def is_fresh(self, max_age_sec: float = 60.0) -> bool:
        """Check if ATR is recent enough for trading decisions"""
        return self.age_seconds <= max_age_sec


class SafeATRProvider:
    """Thread-safe ATR provider with staleness checks"""
    
    def __init__(self, indicator_engine, max_cache_age: float = 60.0):
        self._engine = indicator_engine
        self._max_age = max_cache_age
        self._cache: dict[str, ATRSnapshot] = {}
        self._lock = threading.RLock()
        self._logger = get_logger(__name__)
        # ✅ FIX: Rate limiting for error logs
        self._error_log_times: dict[str, float] = {}
        self._error_log_interval = 60.0  # Log errors once per minute per symbol
    
    def get_atr(self, symbol: str, *, fallback: float | None = None) -> ATRSnapshot | None:
        """
        Fetch ATR with validation and staleness checks.
        
        Args:
            symbol: Trading symbol
            fallback: Optional static fallback value if ATR unavailable
            
        Returns:
            ATRSnapshot if valid, None if stale/unavailable
        """
        with self._lock:
            # 1. Try cache first
            if symbol in self._cache:
                cached = self._cache[symbol]
                if cached.is_fresh(self._max_age):
                    return cached
            
            # 2. Fetch fresh ATR
            try:
                raw_atr = self._engine.compute_atr(symbol)
                if raw_atr is None or raw_atr <= 0 or not math.isfinite(raw_atr):
                    raise ValueError(f"Invalid ATR value: {raw_atr}")
                
                snapshot = ATRSnapshot(
                    value=float(raw_atr),
                    timestamp=time(),
                    source="live"
                )
                self._cache[symbol] = snapshot
                return snapshot
                
            except Exception as exc:
                # ✅ FIX: Rate-limited error logging
                now = time()
                last_log = self._error_log_times.get(symbol, 0)
                
                if now - last_log > self._error_log_interval:
                    self._logger.warning(
                        f"ATR unavailable for {symbol} (will use fallback)",
                        extra={"event": "atr_fetch_warning", "symbol": symbol}
                    )
                    self._error_log_times[symbol] = now
                
                # 3. Use fallback if provided
                if fallback is not None and fallback > 0:
                    return ATRSnapshot(
                        value=fallback,
                        timestamp=time(),
                        source="fallback"
                    )
                
                # 4. ✅ NEW: Try to compute fallback from price
                try:
                    hist = self._engine._histories.get(symbol)
                    if hist:
                        closes = hist.get_closes(1)
                        if closes and closes[0] > 0:
                            estimated = closes[0] * 0.015  # 1.5% of price
                            return ATRSnapshot(
                                value=estimated,
                                timestamp=time(),
                                source="estimated"
                            )
                except Exception:
                    pass
                
                return None
    
    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear cached ATR data."""
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()
                
