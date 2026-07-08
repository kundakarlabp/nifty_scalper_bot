"""Time-bucketed OHLCV helpers for consistent indicator updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass(slots=True)
class OneMinuteBar:
    """Immutable representation of a completed one-minute OHLCV bar."""

    open: float
    high: float
    low: float
    close: float
    volume: int
    start: datetime
    end: datetime
    synthetic: bool = False

    @property
    def timestamp(self) -> datetime:
        """
        Canonical candle timestamp.
        Contract: all consumers must use candle.timestamp
        """
        return self.start

    def as_mapping(self) -> dict[str, Any]:
        """Return a mapping compatible with :class:`PriceHistory.add_tick`.

        Returns:
            dict[str, Any]: Normalised OHLCV payload with ISO timestamps.

        Raises:
            None.
        """

        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "start": self.start,
            "end": self.end,
            "synthetic": self.synthetic,
        }


class OneMinuteBarBuilder:
    """Accumulate ticks into deterministic one-minute OHLCV bars."""

    def __init__(self) -> None:
        """Create a builder with no prior state."""

        self._current: dict[str, Any] | None = None
        self._logger = LOGGER

    def update(
        self, price: float, volume: int, timestamp: datetime
    ) -> OneMinuteBar | None:
        """Update the builder with a new tick.

        Args:
            price: Last traded price for the tick.
            volume: Reported traded volume for the tick.
            timestamp: Exchange timestamp for the tick.

        Returns:
            OneMinuteBar | None: Completed bar when the update closed a minute,
            otherwise ``None`` while the bar is still forming.

        Raises:
            ValueError: If ``price`` is not strictly positive.
        """

        self._logger.debug(
            "Entered OneMinuteBarBuilder.update",
            extra={"event": "one_minute_bar_update"},
        )
        if price <= 0:
            raise ValueError("price must be positive")

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)
        timestamp = timestamp.astimezone(timezone.utc)
        bucket_start = timestamp.replace(second=0, microsecond=0)

        if self._current is None:
            self._current = self._start_bar(price, volume, bucket_start, timestamp)
            return None

        current_bucket = self._current["bucket"]
        if bucket_start < current_bucket:
            self._logger.debug(
                "Dropping late tick for already-active minute bucket",
                extra={
                    "event": "one_minute_bar_late_tick_dropped",
                    "tick_bucket": bucket_start.isoformat(),
                    "current_bucket": current_bucket.isoformat(),
                    "tick_ts": timestamp.isoformat(),
                },
            )
            return None
        if bucket_start == current_bucket:
            self._extend_bar(price, volume, timestamp)
            return None

        completed = self._finalise_current()
        self._current = self._start_bar(price, volume, bucket_start, timestamp)
        return completed

    def flush(self) -> OneMinuteBar | None:
        """Return the in-flight bar if available and reset the builder.

        Returns:
            OneMinuteBar | None: Remaining bar waiting to be flushed.

        Raises:
            None.
        """

        if self._current is None:
            return None
        completed = self._finalise_current()
        self._current = None
        return completed

    # ------------------------------------------------------------------
    def _start_bar(
        self,
        price: float,
        volume: int,
        bucket_start: datetime,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Start a new bar tracking payload for *bucket_start* minute.

        Args:
            price: First traded price captured inside the bucket.
            volume: Volume associated with the trade.
            bucket_start: Normalised minute-aligned timestamp.
            timestamp: Original tick timestamp for audit trail.

        Returns:
            dict[str, Any]: Mutable tracking state for the live bar.

        Raises:
            None.
        """

        self._logger.debug(
            "Condition met: starting new minute bar",
            extra={"event": "one_minute_bar_start", "bucket": bucket_start.isoformat()},
        )
        return {
            "bucket": bucket_start,
            "open": float(price),
            "high": float(price),
            "low": float(price),
            "close": float(price),
            "volume": max(int(volume), 0),
            "start": bucket_start,
            "end": timestamp,
        }

    def _extend_bar(self, price: float, volume: int, timestamp: datetime) -> None:
        """Extend the current in-flight bar with *price* and *volume*.

        Args:
            price: Latest traded price inside the active bucket.
            volume: Trade volume to accumulate.
            timestamp: Tick timestamp associated with the update.

        Returns:
            None.

        Raises:
            None.
        """

        if self._current is None:
            return
        self._logger.debug(
            "Condition met: extending current minute bar",
            extra={"event": "one_minute_bar_extend"},
        )
        self._current["high"] = max(self._current["high"], float(price))
        self._current["low"] = min(self._current["low"], float(price))
        self._current["close"] = float(price)
        self._current["volume"] += max(int(volume), 0)
        self._current["end"] = timestamp

    def _finalise_current(self) -> OneMinuteBar:
        """Convert the tracking payload into an immutable :class:`OneMinuteBar`.

        Args:
            None.

        Returns:
            OneMinuteBar: Completed bar carrying OHLCV aggregates.

        Raises:
            AssertionError: If called without an active bar (defensive).
        """

        if self._current is None:
            raise RuntimeError("Cannot finalise bar without an active payload")
        payload = self._current
        bar = OneMinuteBar(
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=int(payload["volume"]),
            start=payload["start"],
            end=payload["end"],
        )
        self._logger.debug(
            "Condition met: finalised minute bar",
            extra={
                "event": "one_minute_bar_complete",
                "start": bar.start.isoformat(),
                "end": bar.end.isoformat(),
            },
        )
        return bar
