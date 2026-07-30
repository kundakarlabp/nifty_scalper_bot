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


def test_bootstrap_import_preserves_newer_current_candle() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.current_candle = _row(2)

    engine.import_history(pd.DataFrame([_row(0), _row(1)]), mode="bootstrap")

    assert engine.get_current_candle() == _row(2)
    assert len(engine.get_completed_bars()) == 2


def test_bootstrap_equal_minute_overlap_uses_history_authoritative_compatibility() -> (
    None
):
    engine = CandleEngine(symbol="NFO:T")
    engine.current_candle = _row(1, close=100.75)

    engine.import_history(pd.DataFrame([_row(0), _row(1)]), mode="bootstrap")

    assert engine.get_current_candle() is None
    assert float(engine.get_df().iloc[-1]["close"]) == 100.5


def test_bootstrap_rejects_older_current_candle_atomically() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0)]), mode="bootstrap")
    engine.current_candle = _row(1)
    before = engine.get_df()

    with pytest.raises(DataIntegrityError):
        engine.import_history(
            pd.DataFrame([_row(0), _row(1), _row(2)]), mode="bootstrap"
        )

    pd.testing.assert_frame_equal(engine.get_df(), before)
    assert engine.get_current_candle() == _row(1)


def test_empty_bootstrap_import_preserves_current_candle() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0)]), mode="bootstrap")
    engine.current_candle = _row(1)

    engine.import_history(pd.DataFrame(), mode="bootstrap")

    assert engine.get_completed_bars() == []
    assert engine.get_current_candle() == _row(1)


def test_history_import_failure_and_conflict_counters_are_distinct() -> None:
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0)]), mode="bootstrap")

    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([{**_row(1), "high": 99.0}]))
    diagnostics = engine.diagnostics()
    assert diagnostics["history_import_failure_total"] == 1
    assert diagnostics["history_import_conflict_total"] == 0

    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(0, close=100.75)]))
    diagnostics = engine.diagnostics()
    assert diagnostics["history_import_failure_total"] == 2
    assert diagnostics["history_import_conflict_total"] == 1


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


def test_rest_historical_overlap_with_ws_finalized_candle_is_reconciled_not_fatal() -> (
    None
):
    """Production incident (2026-07-20 09:59 IST): a REST historical backfill
    batch spanning the previous session contained the already-finalized WS
    candle at 09:19 with a slightly different close/volume (normal — WS may
    miss early/late ticks; REST is the exchange-finalized aggregate). The old
    behavior aborted the ENTIRE batch (new_ingested_bars=0), leaving canonical
    history stuck below the 50-bar requirement forever.

    With source="historical" declared, one reconciled overlap must not abort
    bars before/after it, and the reconciled bar must take the REST value.
    """
    engine = CandleEngine(symbol="NFO:T")
    # WS-built candle already finalized locally at "09:19" (minute index 0).
    engine.import_history(
        pd.DataFrame([_row(0, close=100.5, volume=10.0)]),
        mode="bootstrap",
    )
    before_reconciled = engine.diagnostics()["history_import_reconciled_total"]

    # REST batch: bars before, the SAME minute with a legitimately different
    # OHLC/volume (exchange-finalized), and valid bars after.
    rest_batch = pd.DataFrame(
        [
            _row(-2), _row(-1),
            _row(0, close=100.65, volume=11.0),  # overlap: reconciled, not fatal
            _row(1), _row(2),
        ]
    )
    engine.import_history(rest_batch, source="historical")

    bars = engine.get_completed_bars()
    timestamps = [b["timestamp"] for b in bars]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)  # strictly unique
    # Batch continued: bars before AND after the overlap were both accepted.
    assert len(bars) == 5
    # The overlapping minute now holds the REST (incoming) value.
    reconciled_row = next(b for b in bars if b["timestamp"] == _ts(0))
    assert reconciled_row["close"] == 100.65
    assert reconciled_row["volume"] == 11.0
    assert (
        engine.diagnostics()["history_import_reconciled_total"]
        == before_reconciled + 1
    )
    # No fatal batch failure was recorded for this reconciled overlap.
    assert engine.diagnostics()["history_import_conflict_total"] == 0


def test_undeclared_source_overlap_still_fails_closed() -> None:
    """Reconciliation is scoped strictly to declared REST-historical sources.
    An incremental import with no declared provenance keeps the original
    strict fail-closed contract (unchanged; see the conflict tests above)."""
    engine = CandleEngine(symbol="NFO:T")
    engine.import_history(pd.DataFrame([_row(0, close=100.5)]), mode="bootstrap")
    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([_row(0, close=100.75)]))  # source=None
    assert engine.diagnostics()["history_import_reconciled_total"] == 0


def test_malformed_and_future_rows_still_rejected_during_reconcilable_import() -> None:
    """Reconciliation never weakens the existing integrity/future-candle
    guards — only the same-timestamp-conflict-vs-existing-store path changes."""
    engine = CandleEngine(symbol="NFO:T")
    with pytest.raises(DataIntegrityError):
        engine.import_history(
            pd.DataFrame([{**_row(0), "high": 99.0}]), source="historical"
        )
    with pytest.raises(DataIntegrityError):
        engine.import_history(
            pd.DataFrame([_row(0, volume=-1.0)]), source="historical"
        )
    future_row = {**_row(0), "timestamp": pd.Timestamp.now(tz=IST) + pd.Timedelta(days=1)}
    with pytest.raises(DataIntegrityError):
        engine.import_history(pd.DataFrame([future_row]), source="historical")
    # Contradictory duplicate timestamps WITHIN the same incoming REST batch
    # remain a genuine source-integrity error, not a reconcilable overlap.
    with pytest.raises(DataIntegrityError):
        engine.import_history(
            pd.DataFrame([_row(0, close=100.5), _row(0, close=100.9)]),
            source="historical",
        )


def test_historical_import_does_not_finalize_current_exchange_minute() -> None:
    engine = CandleEngine(symbol="NFO:T")
    current_minute = pd.Timestamp.now(tz=IST).floor("min")
    previous_minute = current_minute - pd.Timedelta(minutes=1)
    rows = pd.DataFrame(
        [
            {**_row(0), "timestamp": previous_minute},
            {**_row(1), "timestamp": current_minute},
        ]
    )

    engine.import_history(rows, source="historical")

    assert engine.latest_finalized_minute() == previous_minute
    result = engine.on_tick(
        {
            "symbol": "NFO:T",
            "timestamp": current_minute + pd.Timedelta(seconds=30),
            "last_price": 101.0,
            "volume": 1.0,
        }
    )
    assert result is None
    assert engine.get_current_candle()["timestamp"] == current_minute
