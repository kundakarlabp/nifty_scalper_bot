from __future__ import annotations

import pandas as pd

FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def _canonical(frame: pd.DataFrame) -> list[tuple]:
    out = frame[FIELDS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).astype("int64")
    return [tuple(row) for row in out.itertuples(index=False, name=None)]


def _indicator_frame(indicator_engine, symbol: str) -> pd.DataFrame:
    rows = indicator_engine.get_history(symbol, field="bars")
    return pd.DataFrame(rows, columns=FIELDS)


def _runner_frame(runner, symbol: str) -> pd.DataFrame:
    rows = getattr(runner, "_symbol_history", {}).get(symbol, [])  # noqa: SLF001
    if rows:
        return pd.DataFrame(rows, columns=FIELDS)
    return _indicator_frame(
        getattr(runner, "_indicator_engine"), symbol
    )  # noqa: SLF001


def assert_candle_ssot_consistent(
    *, symbol: str, candle_engine, indicator_engine, runner
) -> None:
    deque_rows = pd.DataFrame(
        list(candle_engine._completed_candles), columns=FIELDS
    )  # noqa: SLF001 - test-only SSOT assertion
    engine_rows = candle_engine.get_df().tail(len(deque_rows))
    indicator_rows = _indicator_frame(indicator_engine, symbol).tail(len(deque_rows))
    runner_rows = _runner_frame(runner, symbol).tail(len(deque_rows))
    expected = _canonical(deque_rows)
    assert _canonical(engine_rows) == expected
    assert _canonical(indicator_rows) == expected
    assert len(runner_rows) >= len(expected)
    diag = candle_engine.diagnostics()
    assert diag["candle_store_size"] <= diag["candle_store_maxlen"]
    ts = pd.to_datetime(engine_rows["timestamp"])
    assert ts.is_monotonic_increasing and not ts.duplicated().any()
