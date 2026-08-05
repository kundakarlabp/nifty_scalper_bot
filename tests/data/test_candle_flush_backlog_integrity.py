from __future__ import annotations

import pandas as pd

from nifty_scalper_bot.data.market_data_hardening import _flush_due_candles


class _Engine:
    def __init__(self) -> None:
        self.current_candle = {
            "timestamp": pd.Timestamp("2026-08-05 10:29:00", tz="Asia/Kolkata"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10,
        }
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1
        candle = dict(self.current_candle)
        self.current_candle = None
        return candle


class _Logger:
    def error(self, *args, **kwargs) -> None:
        raise AssertionError(args)

    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass


class _Manager:
    def __init__(self) -> None:
        self._engines = {"NFO:NIFTY2681124600PE": _Engine()}
        self._pending_tick_count = 1
        self._tick_drain_active = 0
        self._last_candle_flush_log_mono = 0.0
        self._logger = _Logger()


def test_clock_flush_waits_for_accepted_tick_backlog() -> None:
    manager = _Manager()
    now = pd.Timestamp("2026-08-05 10:30:05", tz="Asia/Kolkata")

    assert _flush_due_candles(manager, now=now, grace_seconds=1.5) == 0
    assert manager._engines["NFO:NIFTY2681124600PE"].flush_calls == 0

    manager._pending_tick_count = 0
    assert _flush_due_candles(manager, now=now, grace_seconds=1.5) == 1
    assert manager._engines["NFO:NIFTY2681124600PE"].flush_calls == 1


def test_clock_flush_waits_while_tick_drain_is_active() -> None:
    manager = _Manager()
    manager._pending_tick_count = 0
    manager._tick_drain_active = 1
    now = pd.Timestamp("2026-08-05 10:30:05", tz="Asia/Kolkata")

    assert _flush_due_candles(manager, now=now, grace_seconds=1.5) == 0
    assert manager._engines["NFO:NIFTY2681124600PE"].flush_calls == 0
