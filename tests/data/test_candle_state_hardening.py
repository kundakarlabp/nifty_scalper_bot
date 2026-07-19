"""Regression coverage for bootstrap/live candle reconciliation races."""

from __future__ import annotations

import threading

import pandas as pd

from nifty_scalper_bot.data.candle_clock_flush_hardening import (
    install_candle_clock_flush_hardening,
)
from nifty_scalper_bot.data.candle_engine import IST, CandleEngine
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


class _Logger:
    def __init__(self) -> None:
        self.errors: list[tuple[object, ...]] = []

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        self.errors.append(args)

    def debug(self, *args, **kwargs):
        return None


def _manager_for(engine: CandleEngine):
    class Manager:
        pass

    install_candle_clock_flush_hardening(Manager)
    manager = Manager()
    manager._engines = {"NFO:NIFTY26JULFUT": engine}
    manager._candle_flush_grace_s = 1.5
    manager._last_candle_flush_log_mono = 0.0
    manager._lock = threading.RLock()
    manager._ohlc = {"NFO:NIFTY26JULFUT": []}
    manager._logger = _Logger()
    return manager


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
        assert (
            first.on_tick(_tick(minute + pd.Timedelta(seconds=second), 101.0)) is None
        )
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
    assert diagnostics["current_reconcile_total"] == 1
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


def test_next_minute_tick_discards_stale_finalized_current_before_rollover() -> None:
    minute = _minute()
    next_minute = minute + pd.Timedelta(minutes=1)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute, close=100.5)]))
    # Reproduce the production race: hydration has finalized the minute while a
    # live partial for the same minute remains/reappears before the next tick.
    engine.current_candle = {
        "timestamp": minute,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 4.0,
    }

    assert engine.on_tick(_tick(next_minute + pd.Timedelta(seconds=1), 103.0)) is None

    assert engine.current_candle is not None
    assert pd.Timestamp(engine.current_candle["timestamp"]) == next_minute
    assert float(engine.current_candle["open"]) == 103.0
    assert len(engine.get_df()) == 1
    diagnostics = engine.diagnostics()
    assert diagnostics["same_minute_conflict_total"] == 0
    assert diagnostics["current_reconcile_tick_total"] == 1
    assert diagnostics["state_consistent"] is True


def test_direct_flush_discards_stale_finalized_current_without_conflict() -> None:
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

    assert engine.flush() is None
    assert engine.current_candle is None
    assert len(engine.get_df()) == 1
    diagnostics = engine.diagnostics()
    assert diagnostics["same_minute_conflict_total"] == 0
    assert diagnostics["current_reconcile_flush_total"] == 1


def test_clock_flush_discards_stale_finalized_current_without_error_loop() -> None:
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
    manager = _manager_for(engine)

    for second in (2, 4, 6):
        assert (
            manager.flush_due_candles(
                now=minute + pd.Timedelta(minutes=1, seconds=second),
                grace_seconds=1.5,
            )
            == 0
        )

    assert engine.current_candle is None
    assert manager._ohlc["NFO:NIFTY26JULFUT"] == []
    assert manager._logger.errors == []
    diagnostics = engine.diagnostics()
    assert diagnostics["same_minute_conflict_total"] == 0
    assert diagnostics["current_reconcile_total"] == 1


def test_clock_flush_does_not_flush_new_current_minute_after_tick_rollover() -> None:
    minute = _minute()
    next_minute = minute + pd.Timedelta(minutes=1)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.on_tick(_tick(minute + pd.Timedelta(seconds=5), 100.0))
    engine.on_tick(_tick(next_minute + pd.Timedelta(seconds=1), 101.0))
    manager = _manager_for(engine)

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
        assert (
            engine.on_tick(_tick(minute + pd.Timedelta(seconds=second), 101.0)) is None
        )

    assert engine.current_candle is None
    assert engine.diagnostics()["finalized_minute_tick_reject_total"] == 4
    assert engine.is_state_consistent() is True


def test_state_hardening_installer_is_native_idempotent() -> None:
    on_tick = CandleEngine.on_tick
    replace_history = CandleEngine.replace_history
    flush = CandleEngine.flush

    install_candle_state_hardening(CandleEngine)
    install_candle_state_hardening(CandleEngine)

    assert CandleEngine.on_tick is on_tick
    assert CandleEngine.replace_history is replace_history
    assert CandleEngine.flush is flush

    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute)]))
    engine.on_tick(_tick(minute + pd.Timedelta(seconds=5), 101.0))
    assert engine.diagnostics()["finalized_minute_tick_reject_total"] == 1


def test_native_reconciliation_raises_for_current_older_than_finalized() -> None:
    from nifty_scalper_bot.data.source import DataIntegrityError

    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_history_row(minute)]))
    engine.current_candle = _history_row(minute - pd.Timedelta(minutes=1))

    before = engine.get_current_candle()
    try:
        engine.reconcile_current_with_finalized(reason="flush")
    except DataIntegrityError:
        pass
    else:
        raise AssertionError("older current must fail closed")

    assert engine.get_current_candle() == before
    assert engine.is_state_consistent() is False


def test_state_consistency_detects_duplicate_and_non_monotonic_completed_history() -> (
    None
):
    from collections import deque

    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine._completed_candles = deque(
        [_history_row(minute), _history_row(minute)], maxlen=engine.max_bars
    )
    assert engine.is_state_consistent() is False

    engine._completed_candles = deque(
        [_history_row(minute), _history_row(minute - pd.Timedelta(minutes=1))],
        maxlen=engine.max_bars,
    )
    assert engine.is_state_consistent() is False


def test_bounded_concurrent_native_state_transitions_do_not_corrupt_state() -> None:
    minute = _minute()
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.on_tick(_tick(minute + pd.Timedelta(seconds=1), 100.0))
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def run(action):
        try:
            barrier.wait(timeout=5)
            action()
        except BaseException as exc:  # pragma: no cover - failure captured below
            errors.append(exc)

    threads = [
        threading.Thread(
            target=run,
            args=(
                lambda: engine.on_tick(
                    _tick(minute + pd.Timedelta(minutes=1, seconds=1), 101.0)
                ),
            ),
        ),
        threading.Thread(
            target=run,
            args=(
                lambda: engine.import_history(
                    pd.DataFrame(
                        [
                            {
                                "timestamp": minute,
                                "open": 100.0,
                                "high": 100.0,
                                "low": 100.0,
                                "close": 100.0,
                                "volume": 1.0,
                            }
                        ]
                    )
                ),
            ),
        ),
        threading.Thread(target=run, args=(engine.flush,)),
        threading.Thread(target=run, args=(engine.diagnostics,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    timestamps = [row["timestamp"] for row in engine.get_completed_bars()]
    assert len(timestamps) == len(set(timestamps))
    assert engine.is_state_consistent() is True
