"""Regression coverage for bootstrap/live candle reconciliation races."""

from __future__ import annotations

import pandas as pd
import pytest

from nifty_scalper_bot.data.candle_clock_flush_hardening import (
    install_candle_clock_flush_hardening,
)
from nifty_scalper_bot.data.candle_engine import CandleEngine, DataIntegrityError, IST
from nifty_scalper_bot.data.candle_state_hardening import install_candle_state_hardening

install_candle_state_hardening(CandleEngine)


def _minute(offset: int = 0) -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST).floor("min") - pd.Timedelta(minutes=10 - offset)


def _history_row(timestamp: pd.Timestamp, close: float = 100.5) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": 100.0,
        "high": max(101.0, close),
        "low": 99.0,
        "close": close,
        "volume": 10.0,
    }


def _tick(timestamp: pd.Timestamp, price: float = 100.0) -> dict[str, object]:
    return {
        "symbol": "NFO:NIFTY26JULFUT",
        "timestamp": timestamp,
        "ltp": price,
        "volume": 1.0,
    }


def test_delayed_tick_cannot_reopen_finalized_minute() -> None:
    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute)]))

    assert engine.on_tick(_tick(minute + pd.Timedelta(seconds=45), 101.0)) is None
    assert engine.current_candle is None
    assert len(engine.get_df()) == 1
    diagnostics = engine.diagnostics()
    assert diagnostics["finalized_minute_tick_reject_total"] == 1
    assert diagnostics["state_consistent"] is True


def test_new_engine_does_not_inherit_hardening_diagnostics() -> None:
    minute = _minute()
    first = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    first.replace_history(pd.DataFrame([_history_row(minute)]))
    for second in (5, 15, 30):
        assert first.on_tick(_tick(minute + pd.Timedelta(seconds=second), 101.0)) is None
    assert first.diagnostics()["finalized_minute_tick_reject_total"] == 3

    second = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    assert second.diagnostics()["finalized_minute_tick_reject_total"] == 0
    assert second.diagnostics()["history_current_reconcile_total"] == 0


def test_history_replacement_discards_overlapping_partial_candle() -> None:
    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.current_candle = {
        "timestamp": minute,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 4.0,
    }

    engine.replace_history(pd.DataFrame([_history_row(minute, close=100.5)]))

    assert engine.current_candle is None
    assert float(engine.get_df().iloc[-1]["close"]) == 100.5
    diagnostics = engine.diagnostics()
    assert diagnostics["history_current_reconcile_total"] == 1
    assert diagnostics["state_consistent"] is True


def test_history_older_than_current_candle_preserves_live_minute() -> None:
    history_minute = _minute()
    current_minute = history_minute + pd.Timedelta(minutes=1)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.current_candle = {
        "timestamp": current_minute,
        "open": 101.0,
        "high": 101.0,
        "low": 101.0,
        "close": 101.0,
        "volume": 1.0,
    }

    engine.replace_history(pd.DataFrame([_history_row(history_minute)]))

    assert engine.current_candle is not None
    assert pd.Timestamp(engine.current_candle["timestamp"]) == current_minute
    assert engine.is_state_consistent() is True


def test_conflict_clears_poisoned_current_candle_once() -> None:
    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute, close=100.5)]))
    engine.current_candle = {
        "timestamp": minute,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 4.0,
    }

    with pytest.raises(DataIntegrityError):
        engine.flush()

    assert engine.current_candle is None
    assert engine.flush() is None
    assert len(engine.get_df()) == 1
    assert engine.diagnostics()["same_minute_conflict_total"] == 1


def test_clock_flush_does_not_flush_new_current_minute_after_tick_rollover() -> None:
    minute = _minute()
    next_minute = minute + pd.Timedelta(minutes=1)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.on_tick(_tick(minute + pd.Timedelta(seconds=5), 100.0))
    engine.on_tick(_tick(next_minute + pd.Timedelta(seconds=1), 101.0))

    class Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    class Manager:
        pass

    install_candle_clock_flush_hardening(Manager)
    manager = Manager()
    manager._engines = {"NFO:NIFTY26JULFUT": engine}
    manager._candle_flush_grace_s = 1.5
    manager._last_candle_flush_log_mono = 0.0
    manager._lock = __import__("threading").RLock()
    manager._ohlc = {"NFO:NIFTY26JULFUT": []}
    manager._logger = Logger()

    flushed = manager.flush_due_candles(
        now=minute + pd.Timedelta(minutes=1, seconds=2),
        grace_seconds=1.5,
    )

    assert flushed == 0
    assert engine.current_candle is not None
    assert pd.Timestamp(engine.current_candle["timestamp"]) == next_minute
    assert len(engine.get_df()) == 1


def test_replayed_old_ticks_after_reconciliation_do_not_recreate_conflict() -> None:
    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute)]))

    for second in (5, 15, 30, 45):
        assert engine.on_tick(_tick(minute + pd.Timedelta(seconds=second), 101.0)) is None

    assert engine.current_candle is None
    assert engine.diagnostics()["finalized_minute_tick_reject_total"] == 4
    assert engine.is_state_consistent() is True
