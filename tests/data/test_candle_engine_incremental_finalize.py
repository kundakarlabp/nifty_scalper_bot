"""Live candle finalization must be incremental: no full-frame sanitize."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from nifty_scalper_bot.data import candle_engine as ce_mod
from nifty_scalper_bot.data.candle_engine import IST, CandleEngine, DataIntegrityError


def _mk(ts, o=100.0, h=101.0, low=99.5, c=100.5, v=10.0):
    return {"timestamp": ts, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _ts(minute: int) -> pd.Timestamp:
    base = pd.Timestamp.now(tz=IST).floor("min") - pd.Timedelta(minutes=60)
    return base + pd.Timedelta(minutes=minute)


def _finalize(engine: CandleEngine, candle: dict):
    engine.current_candle = candle
    return engine._finalize_current_candle()


def test_incremental_append_correct_and_bounded(monkeypatch) -> None:
    calls = []
    real = ce_mod.sanitize
    monkeypatch.setattr(ce_mod, "sanitize", lambda f: calls.append(1) or real(f))

    engine = CandleEngine(symbol="NFO:T", max_bars=5)
    assert _finalize(engine, _mk(_ts(0))) is not None      # first append
    assert _finalize(engine, _mk(_ts(1))) is not None      # later append
    assert calls == [], "live finalize must NOT call full sanitize"

    # same-minute duplicate: no duplicate timestamp row
    assert _finalize(engine, _mk(_ts(1))) is None
    # out-of-order rejected
    with pytest.raises(DataIntegrityError):
        _finalize(engine, _mk(_ts(0)))
    # invalid OHLC rejected (existing contract: raises via _validate_ohlc_row)
    with pytest.raises(DataIntegrityError):
        _finalize(engine, _mk(_ts(2), o=105.0, h=101.0))
    engine.current_candle = None
    # negative volume rejected (raise or silent-drop are both fail-closed)
    try:
        assert _finalize(engine, _mk(_ts(3), v=-5.0)) is None
    except DataIntegrityError:
        pass
    engine.current_candle = None
    # future candle rejected
    assert _finalize(engine, _mk(pd.Timestamp.now(tz=IST) + pd.Timedelta(minutes=10))) is None

    for minute in range(2, 12):  # bound to max_bars
        _finalize(engine, _mk(_ts(minute)))
    df = engine.get_df()
    assert len(df) <= 5
    ts = pd.to_datetime(df["timestamp"])
    assert ts.is_monotonic_increasing and not ts.duplicated().any()
    assert str(ts.iloc[-1].tz) == "Asia/Kolkata"
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert calls == []


def test_market_open_rollover_burst_keeps_loop_responsive() -> None:
    engines = [CandleEngine(symbol=f"NFO:S{i}", max_bars=600) for i in range(20)]
    for engine in engines:  # 500 stored bars each
        rows = [_mk(_ts(-600 + m)) for m in range(500)]
        engine.df = pd.DataFrame(rows)

    async def scenario() -> float:
        beats: list[float] = []

        async def heartbeat():
            for _ in range(10):
                start = time.monotonic()
                await asyncio.sleep(0.005)
                beats.append(time.monotonic() - start)

        async def rollover():
            for minute in range(3):  # simultaneous minute boundaries
                for engine in engines:
                    _finalize(engine, _mk(_ts(-100 + minute)))
                await asyncio.sleep(0)

        await asyncio.gather(heartbeat(), rollover())
        return max(beats)

    worst = asyncio.run(scenario())
    for engine in engines:
        df = engine.get_df()
        ts = pd.to_datetime(df["timestamp"])
        assert not ts.duplicated().any() and ts.is_monotonic_increasing
    # Cooperative bound: heartbeat scheduling delay stays far below the
    # 500-850ms production lag this fix targets.
    assert worst < 0.25, worst
