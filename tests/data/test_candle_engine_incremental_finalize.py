"""Live candle finalization must be incremental: no full-frame sanitize."""

from __future__ import annotations

import asyncio
import time

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
    assert _finalize(engine, _mk(_ts(0))) is not None  # first append
    assert _finalize(engine, _mk(_ts(1))) is not None  # later append
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
    assert (
        _finalize(engine, _mk(pd.Timestamp.now(tz=IST) + pd.Timedelta(minutes=10)))
        is None
    )

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


def test_live_finalize_uses_deque_not_dataframe_append_paths(monkeypatch) -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=3)
    concat_calls = []
    monkeypatch.setattr(
        pd,
        "concat",
        lambda *a, **k: concat_calls.append(1)
        or pytest.fail("pd.concat used on live append"),
    )
    result = _finalize(engine, _mk(_ts(20)))
    assert result is not None
    assert concat_calls == []
    assert engine.diagnostics()["candle_store_size"] == 1
    assert engine.diagnostics()["live_append_total"] == 1


def test_returned_candle_is_canonical_and_cache_rebuilds_only_when_dirty() -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=5)
    candle = {
        "timestamp": _ts(30),
        "open": "100.0",
        "high": "102.0",
        "low": "99.0",
        "close": "101.0",
        "volume": "2500",
    }
    returned = _finalize(engine, candle)
    stored = engine.get_df().iloc[-1].to_dict()
    for field in ("open", "high", "low", "close", "volume"):
        assert isinstance(returned[field], float)
        assert isinstance(stored[field], float)
        assert returned[field] == stored[field]
    assert returned["timestamp"] == stored["timestamp"]
    diag_after_first_read = engine.diagnostics()
    engine.get_df()
    assert (
        engine.diagnostics()["df_cache_rebuild_total"]
        == diag_after_first_read["df_cache_rebuild_total"]
    )
    _finalize(engine, _mk(_ts(31)))
    assert engine.diagnostics()["df_cache_dirty"] is True
    engine.get_df()
    assert (
        engine.diagnostics()["df_cache_rebuild_total"]
        == diag_after_first_read["df_cache_rebuild_total"] + 1
    )


def test_same_minute_identical_ignored_and_conflicting_rejected() -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=5)
    assert _finalize(engine, _mk(_ts(40))) is not None
    assert _finalize(engine, _mk(_ts(40))) is None
    assert engine.diagnostics()["same_minute_idempotent_total"] == 1
    with pytest.raises(DataIntegrityError):
        _finalize(engine, _mk(_ts(40), c=100.75))
    assert engine.diagnostics()["same_minute_conflict_total"] == 1


def test_df_getter_is_defensive_copy_and_replacement_methods_work() -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=5)
    first = pd.DataFrame([_mk(_ts(50), c=100.5)])
    engine.df = first
    copy_frame = engine.df
    copy_frame.loc[0, "close"] = 999.0
    copy_frame["extra"] = "ignored"
    stored = engine.get_df()
    assert float(stored.loc[0, "close"]) == 100.5
    assert "extra" not in stored.columns

    assigned = pd.DataFrame([_mk(_ts(51), h=103.0, c=101.5)])
    engine.df = assigned
    assert float(engine.get_df().loc[0, "close"]) == 101.5

    replaced = pd.DataFrame([_mk(_ts(52), h=104.0, c=102.5)])
    engine.replace_history(replaced)
    assert float(engine.get_df().loc[0, "close"]) == 102.5


def test_candle_engine_df_callers_do_not_use_in_place_mutation() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    production = list((root / "src" / "nifty_scalper_bot").rglob("*.py"))
    offenders: list[str] = []
    for path in production:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"loc", "iloc"}:
                parent = getattr(node, "parent", None)
                if (
                    isinstance(parent, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "df"
                ):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "df"
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg == "inplace"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        text = ast.get_source_segment(source, node) or ""
                        if ".df" in text:
                            offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
