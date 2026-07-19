from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from nifty_scalper_bot.data.candle_engine import IST, CandleEngine, DataIntegrityError


def _ts(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-02T09:15:00Z") + pd.Timedelta(minutes=minute)


def _row(minute: int, *, close: float = 100.5, volume: float = 10.0) -> dict:
    return {
        "timestamp": _ts(minute),
        "open": 100.0,
        "high": max(101.0, close),
        "low": min(99.0, close),
        "close": close,
        "volume": volume,
    }


def test_empty_engine_returns_no_finalized_watermark() -> None:
    engine = CandleEngine(symbol="NFO:T")
    assert engine.latest_finalized_minute() is None


def test_bootstrap_and_incremental_history_import_populate_canonical_history() -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=5)
    engine.import_history(pd.DataFrame([_row(0), _row(1)]), mode="bootstrap")
    engine.import_history(pd.DataFrame([_row(2)]))

    bars = engine.get_completed_bars()
    assert len(bars) == 3
    assert bars[-1]["timestamp"].isoformat() == "2026-01-02T14:47:00+05:30"
    assert engine.latest_finalized_minute() == bars[-1]["timestamp"]


def test_identical_finalized_import_is_idempotent_and_conflict_is_atomic() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0), _row(1)]), mode="bootstrap")
    before = engine.get_df()

    engine.import_history(pd.DataFrame([_row(1)]))
    assert len(engine.get_df()) == 2
    assert engine.diagnostics()["history_import_idempotent_total"] == 1

    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(1, close=100.75)]))
    pd.testing.assert_frame_equal(engine.get_df(), before)
    assert engine.diagnostics()["history_import_conflict_total"] >= 1


def test_invalid_ohlc_negative_volume_and_conflicting_duplicates_rejected() -> None:
    engine = CandleEngine(symbol="NFO:T")

    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([{**_row(0), "high": 99.0}]))
    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(0, volume=-1.0)]))
    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(0), _row(0, close=100.75)]))


def test_identical_duplicate_timestamp_is_accepted_once() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0), _row(0)]))
    assert len(engine.get_completed_bars()) == 1


def test_current_candle_overlap_rules_are_atomic() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.current_candle = _row(2)
    engine.import_history(pd.DataFrame([_row(0), _row(1)]))
    assert engine.get_current_candle() is not None

    engine.import_history(pd.DataFrame([_row(2)]))
    assert engine.get_current_candle() is None

    engine.current_candle = _row(4, close=100.5)
    before = engine.get_df()
    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(3), _row(4, close=100.75)]))
    pd.testing.assert_frame_equal(engine.get_df(), before)
    assert engine.get_current_candle() is not None

    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(5)]))


def test_max_bars_and_defensive_readers_are_enforced() -> None:
    engine = CandleEngine(symbol="NFO:T", max_bars=3)
    engine.import_history(pd.DataFrame([_row(i) for i in range(5)]), mode="bootstrap")
    assert len(engine.get_completed_bars()) == 3

    bars = engine.get_completed_bars()
    bars[-1]["close"] = 999.0
    assert engine.get_completed_bars()[-1]["close"] != 999.0

    frame = engine.get_df()
    frame.loc[frame.index[-1], "close"] = 999.0
    assert float(engine.get_df().iloc[-1]["close"]) != 999.0

    engine.current_candle = _row(6)
    current = engine.get_current_candle()
    assert current is not None
    current["close"] = 999.0
    assert engine.get_current_candle()["close"] != 999.0  # type: ignore[index]


def test_late_tick_at_finalized_watermark_rejected_after_hydration() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0)]), mode="bootstrap")
    assert (
        engine.on_tick({"symbol": "NFO:T", "timestamp": _ts(0), "ltp": 101.0}) is None
    )
    assert engine.diagnostics()["finalized_minute_tick_reject_total"] == 1


def test_history_import_normalizes_timezone_and_does_not_touch_trading_modules() -> (
    None
):
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0)]), mode="bootstrap")
    assert str(engine.get_df().iloc[0]["timestamp"].tz) == str(IST)

    root = Path(__file__).resolve().parents[2]
    forbidden_roots = ("strategies", "execution", "risk")
    offenders: list[str] = []
    for path in (root / "src" / "nifty_scalper_bot").rglob("*.py"):
        rel = path.relative_to(root / "src" / "nifty_scalper_bot")
        if rel.parts and rel.parts[0] in forbidden_roots:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "import_history":
                    offenders.append(str(rel))
    assert offenders == []
