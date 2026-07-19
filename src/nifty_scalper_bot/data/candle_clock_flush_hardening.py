"""Race-safe clock finalization for MarketDataManager candle engines."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.data.candle_state_hardening import reconcile_stale_current

_IST_TZ = "Asia/Kolkata"
_INSTALLED_ATTR = "_candle_clock_flush_hardening_installed"
_LAST_PUBLISHED: dict[int, dict[str, pd.Timestamp]] = defaultdict(dict)


def _coerce_ist_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(_IST_TZ)
    return timestamp.tz_convert(_IST_TZ)


def install_candle_clock_flush_hardening(manager_cls: type[Any]) -> None:
    """Replace clock flush with an expected-minute, race-safe implementation."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    def flush_due_candles(
        self: Any,
        *,
        now: Any | None = None,
        grace_seconds: float | None = None,
    ) -> int:
        grace = (
            max(float(grace_seconds), 0.0)
            if grace_seconds is not None
            else max(float(getattr(self, "_candle_flush_grace_s", 1.5) or 1.5), 0.0)
        )
        now_ts = (
            _coerce_ist_timestamp(now)
            if now is not None
            else pd.Timestamp.now(tz=_IST_TZ)
        )
        if now_ts is None:
            return 0

        engines = list(getattr(self, "_engines", {}).items())
        flushed = 0
        for symbol, engine in engines:
            # Snapshot is only a cheap pre-filter. The same minute is rechecked
            # under the per-engine lock immediately before finalization.
            current = getattr(engine, "current_candle", None)
            if not isinstance(current, Mapping):
                continue
            expected_minute = _coerce_ist_timestamp(current.get("timestamp"))
            if expected_minute is None:
                continue
            if now_ts < expected_minute + pd.Timedelta(minutes=1, seconds=grace):
                continue

            # Completed history is authoritative. Hydration can race with a
            # live partial candle and leave current_candle on an already
            # finalized minute; delegate that reconciliation to CandleEngine.
            if reconcile_stale_current(engine, reason="clock_flush"):
                continue
            locked_current = getattr(engine, "current_candle", None)
            if not isinstance(locked_current, Mapping):
                continue
            locked_minute = _coerce_ist_timestamp(locked_current.get("timestamp"))
            if locked_minute is None or locked_minute != expected_minute:
                # Tick-driven rollover won the race. Never flush the newly
                # opened minute based on the stale pre-lock due decision.
                continue
            if now_ts < locked_minute + pd.Timedelta(minutes=1, seconds=grace):
                continue
            try:
                candle = engine.flush()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "MDM_CANDLE_CLOCK_FLUSH_FAILED symbol=%s error=%r",
                    symbol,
                    exc,
                    exc_info=True,
                    extra={
                        "event": "MDM_CANDLE_CLOCK_FLUSH_FAILED",
                        "symbol": symbol,
                        "error": repr(exc),
                    },
                )
                continue

            if not candle:
                continue
            timestamp = _coerce_ist_timestamp(
                candle["timestamp"] if isinstance(candle, dict) else candle.timestamp
            )
            if timestamp is None:
                continue

            published = _LAST_PUBLISHED[id(self)]
            previous = published.get(symbol)
            if previous is not None and timestamp <= previous:
                self._logger.warning(
                    (
                        "MDM_CANDLE_DUPLICATE_PUBLISH_SUPPRESSED "
                        "symbol=%s timestamp=%s previous=%s"
                    ),
                    symbol,
                    timestamp.isoformat(),
                    previous.isoformat(),
                    extra={
                        "event": "MDM_CANDLE_DUPLICATE_PUBLISH_SUPPRESSED",
                        "symbol": symbol,
                        "timestamp": timestamp.isoformat(),
                        "previous_timestamp": previous.isoformat(),
                    },
                )
                continue

            bar = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": float(
                    candle["open"] if isinstance(candle, dict) else candle.open
                ),
                "high": float(
                    candle["high"] if isinstance(candle, dict) else candle.high
                ),
                "low": float(candle["low"] if isinstance(candle, dict) else candle.low),
                "close": float(
                    candle["close"] if isinstance(candle, dict) else candle.close
                ),
                "volume": int(
                    float(
                        (
                            candle.get("volume", 0)
                            if isinstance(candle, dict)
                            else getattr(candle, "volume", 0)
                        )
                        or 0
                    )
                ),
                "source": "clock_flush_candle",
            }
            with self._lock:
                self._ohlc[symbol].append(bar)
                published[symbol] = timestamp

            publisher = getattr(self, "_publish_closed_bar", None)
            if callable(publisher):
                try:
                    publisher(bar)
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("clock-flush bar publish skipped: %s", exc)

            flushed += 1
            now_mono = time.monotonic()
            if (
                now_mono
                - float(getattr(self, "_last_candle_flush_log_mono", 0.0) or 0.0)
                >= 10.0
            ):
                self._last_candle_flush_log_mono = now_mono
                self._logger.info(
                    "MDM_CANDLE_CLOCK_FLUSHED symbol=%s close=%s",
                    symbol,
                    bar["close"],
                    extra={
                        "event": "MDM_CANDLE_CLOCK_FLUSHED",
                        "symbol": symbol,
                        "source": "clock_flush_candle",
                    },
                )
        return flushed

    manager_cls.flush_due_candles = flush_due_candles
    setattr(manager_cls, _INSTALLED_ATTR, True)


__all__ = ["install_candle_clock_flush_hardening"]
