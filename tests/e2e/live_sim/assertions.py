from __future__ import annotations

import pandas as pd

FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def _canonical(frame: pd.DataFrame) -> list[tuple]:
    out = frame[FIELDS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype(str)
    return [tuple(row) for row in out.itertuples(index=False, name=None)]


def assert_candle_ssot_consistent(
    *, symbol: str, candle_engine, indicator_engine, runner
) -> None:
    deque_rows = pd.DataFrame(
        list(candle_engine._completed_candles), columns=FIELDS
    )  # noqa: SLF001 - test-only SSOT assertion
    engine_rows = candle_engine.get_df().tail(len(deque_rows))
    indicator_rows = indicator_engine.history[symbol].tail(len(deque_rows))
    runner_rows = runner.history[symbol].tail(len(deque_rows))
    expected = _canonical(deque_rows)
    assert _canonical(engine_rows) == expected
    assert _canonical(indicator_rows) == expected
    assert _canonical(runner_rows) == expected
    diag = candle_engine.diagnostics()
    assert diag["candle_store_size"] <= diag["candle_store_maxlen"]
    ts = pd.to_datetime(engine_rows["timestamp"])
    assert ts.is_monotonic_increasing and not ts.duplicated().any()
