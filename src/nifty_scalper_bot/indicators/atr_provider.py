# src/nifty_scalper_bot/indicators/atr_provider.py

import math
import re
import threading
from dataclasses import dataclass
from time import time

from nifty_scalper_bot.utils.logging import get_logger


@dataclass
class ATRSnapshot:
    """Validated ATR data with metadata"""

    value: float
    timestamp: float
    period: int = 14
    source: str = "indicator_engine"

    @property
    def current_atr(self) -> float:
        """Return ATR value for compatibility with legacy callers."""
        return self.value

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
        self._is_warmed_up = False

    @property
    def is_warmed_up(self) -> bool:
        """Return True if indicators are ready for evaluation."""
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        """Mark the provider as ready for trading."""
        self._is_warmed_up = True
        self._logger.info("✅ SafeATRProvider: Indicators marked as WARMED UP.")

    def get_atr(
        self, symbol: str, *, fallback: float | None = None
    ) -> ATRSnapshot | None:
        """Args: symbol, fallback. Returns: ATRSnapshot or None. Raises: None."""
        with self._lock:
            # 1. Try cache first
            if symbol in self._cache:
                cached = self._cache[symbol]
                if cached.is_fresh(self._max_age):
                    return cached

            # 2. Fetch fresh ATR
            try:
                raw_atr = self._engine.compute_atr(symbol)
                snapshot = self._normalize_snapshot(raw_atr, "live")
                if snapshot is None:
                    raise ValueError(f"Invalid ATR value: {raw_atr}")
                self._cache[symbol] = snapshot
                return snapshot

            except Exception:
                self._logger.debug(
                    "atr_compute_failed",
                    extra={"event": "atr_compute_failed", "symbol": symbol},
                    exc_info=True,
                )
                base_symbol = self._resolve_underlying_symbol(symbol)
                if base_symbol and base_symbol != symbol:
                    try:
                        raw_atr = self._engine.compute_atr(base_symbol)
                        snapshot = self._normalize_snapshot(raw_atr, "underlying")
                        if snapshot is not None:
                            self._cache[symbol] = snapshot
                            return snapshot
                    except Exception:
                        self._logger.debug(
                            "atr_underlying_compute_failed",
                            extra={
                                "event": "atr_underlying_compute_failed",
                                "symbol": symbol,
                                "base_symbol": base_symbol,
                            },
                            exc_info=True,
                        )
                # ✅ FIX: Rate-limited error logging
                now = time()
                last_log = self._error_log_times.get(symbol, 0)

                if now - last_log > self._error_log_interval:
                    self._logger.warning(
                        f"ATR unavailable for {symbol} (will use fallback)",
                        extra={"event": "atr_fetch_warning", "symbol": symbol},
                    )
                    self._error_log_times[symbol] = now

                # 3. Use fallback if provided
                if fallback is not None and fallback > 0:
                    return ATRSnapshot(
                        value=fallback,
                        timestamp=time(),
                        source="fallback",
                    )

                # 4. ✅ NEW: Try to compute fallback from price
                try:
                    hist = self._resolve_history(symbol)
                    if hist:
                        closes = hist.get_closes(1)
                        if closes and closes[0] > 0:
                            estimated = closes[0] * 0.015  # 1.5% of price
                            if estimated > 0 and math.isfinite(estimated):
                                return ATRSnapshot(
                                    value=estimated,
                                    timestamp=time(),
                                    source="estimated",
                                )
                except Exception as e:
                    __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
                    raise

                # 5. Final non-None fallback to keep risk guards operational
                return ATRSnapshot(
                    value=5.0,
                    timestamp=time(),
                    source="hardcoded_fallback",
                )

    def _normalize_snapshot(self, raw_atr: object, source: str) -> ATRSnapshot | None:
        """Convert float or ATR-like objects to ATRSnapshot for stable interface."""
        if isinstance(raw_atr, ATRSnapshot):
            return raw_atr
        atr_value = getattr(raw_atr, "current_atr", None)
        if atr_value is None:
            atr_value = getattr(raw_atr, "value", raw_atr)
        try:
            value = float(atr_value)
        except (TypeError, ValueError):
            return None
        if value <= 0 or not math.isfinite(value):
            return None
        return ATRSnapshot(value=value, timestamp=time(), source=source)

    def get_current_atr(self, symbol: str) -> float:
        """Args: symbol. Returns: current ATR value. Raises: None."""
        snapshot = self.get_atr(symbol)
        if snapshot and snapshot.value > 0:
            return float(snapshot.value)
        return 5.0

    def _resolve_underlying_symbol(self, symbol: str) -> str:
        """Args: symbol. Returns: Underlying symbol. Raises: None."""
        try:
            if not symbol:
                return ""
            if symbol.startswith("NIFTY") and not symbol.startswith("NIFTYFUT"):
                return "NIFTY"
            if symbol.startswith("BANKNIFTY"):
                return "BANKNIFTY"
            if symbol.startswith("FINNIFTY"):
                return "FINNIFTY"
            match = re.match(r"^([A-Z]+)", symbol)
            if match:
                return match.group(1)
            return symbol
        except Exception:
            self._logger.debug(
                "atr_underlying_resolve_failed",
                extra={"event": "atr_underlying_resolve_failed", "symbol": symbol},
                exc_info=True,
            )
            return symbol or ""

    def _resolve_history(self, symbol: str) -> object | None:
        """Args: symbol. Returns: Price history or None. Raises: None."""
        try:
            hist = self._engine._histories.get(symbol)
            if hist:
                return hist
            base_symbol = self._resolve_underlying_symbol(symbol)
            if base_symbol and base_symbol != symbol:
                return self._engine._histories.get(base_symbol)
        except Exception:
            self._logger.debug(
                "atr_history_lookup_failed",
                extra={"event": "atr_history_lookup_failed", "symbol": symbol},
                exc_info=True,
            )
        return None

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear cached ATR data."""
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()
