"""Strategy ABC for NiftyScalperBot v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..indicators.engine import IndicatorSnapshot
    from ..regime.detector import MarketRegime
    from ..config.settings import StrategySettings
    from .signal import Signal


class Strategy(ABC):
    """Abstract base for all 12 strategies."""

    name: str = "base"

    def __init__(self, settings: "StrategySettings") -> None:
        self._settings = settings
        self._last_signal_at: datetime | None = None
        self._daily_signal_count: int = 0

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        indicators: "IndicatorSnapshot",
        regime: "MarketRegime",
    ) -> "Signal | None":
        """Evaluate indicators and return a Signal or None."""
        ...

    def _check_cooldown(self) -> bool:
        """True if cooldown has elapsed (OK to signal)."""
        if self._last_signal_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()
        return elapsed >= self._settings.cooldown_sec

    def _check_daily_limit(self) -> bool:
        return self._daily_signal_count < self._settings.max_daily_signals

    def _record_signal(self) -> None:
        self._last_signal_at = datetime.now(timezone.utc)
        self._daily_signal_count += 1

    def reset_daily(self) -> None:
        self._daily_signal_count = 0
        self._last_signal_at = None

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def min_confidence(self) -> float:
        return self._settings.min_confidence

    @property
    def daily_signal_count(self) -> int:
        return self._daily_signal_count

    @property
    def last_signal_at(self) -> datetime | None:
        return self._last_signal_at
