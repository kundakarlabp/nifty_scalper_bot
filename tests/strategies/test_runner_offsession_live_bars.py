"""Runner candle session integrity must not interrupt LIVE quote/protection flow."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from nifty_scalper_bot.strategies.bar_builder import (
    OneMinuteBar,
    OneMinuteBarBuilder,
)
from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    _update_runner_live_candle,
)

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NFO:NIFTY26O1923150CE"


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 10, 16, hour, minute, tzinfo=IST)


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.delenv("NSB_TEST_MODE", raising=False)
    monkeypatch.delenv("BROKER_SIMULATION", raising=False)


def test_offsession_tick_does_not_start_or_advance_runner_builder(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    builder = OneMinuteBarBuilder()
    assert _update_runner_live_candle(builder, SYMBOL, 100.0, 10, _at(18, 46)) is None
    assert builder._current is None

    assert _update_runner_live_candle(builder, SYMBOL, 101.0, 10, _at(15, 29)) is None
    previous = dict(builder._current or {})
    assert previous["bucket"] == _at(15, 29).astimezone(timezone.utc)
    assert _update_runner_live_candle(builder, SYMBOL, 200.0, 10, _at(18, 47)) is None
    assert builder._current == previous


def test_in_session_minute_rollover_keeps_runner_candles(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    builder = OneMinuteBarBuilder()
    assert _update_runner_live_candle(builder, SYMBOL, 100.0, 10, _at(10, 15)) is None
    completed = _update_runner_live_candle(builder, SYMBOL, 101.0, 10, _at(10, 16))
    assert completed is not None
    assert completed.close == 100.0
    assert completed.timestamp == _at(10, 15).astimezone(timezone.utc)


def test_shadow_replay_can_still_build_afterhours_bars(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    builder = OneMinuteBarBuilder()
    assert _update_runner_live_candle(builder, SYMBOL, 100.0, 10, _at(18, 46)) is None
    assert _update_runner_live_candle(builder, SYMBOL, 101.0, 10, _at(18, 47))


def test_nonbackfill_offsession_bar_cannot_advance_indicator_history(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._last_bar_ts = {}
    runner._symbol_history = {}
    runner._indicator_engine = SimpleNamespace(
        update_bar=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("indicator must not receive offsession bar")
        )
    )
    stamp = _at(18, 46).astimezone(timezone.utc)
    bar = OneMinuteBar(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=10,
        start=stamp,
        end=stamp + timedelta(minutes=1),
    )
    runner._ingest_bar(SYMBOL, bar)
    assert runner._symbol_history == {}
    assert runner._last_bar_ts == {}


def test_runner_live_bar_guard_is_inside_phase4_not_before_protection() -> None:
    source = inspect.getsource(StrategyRunner._on_tick)
    assert source.index("_handle_position_tick_protection") < source.index(
        "_update_runner_live_candle("
    )
