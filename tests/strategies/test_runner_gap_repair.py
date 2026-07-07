from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _bar(ts: datetime, close: float) -> OneMinuteBar:
    return OneMinuteBar(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=10,
        start=ts,
        end=ts + timedelta(seconds=59),
        synthetic=False,
    )


def test_gap_repair_skips_small_option_gap() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._main_loop = None
    runner._gap_repair_inflight = set()
    runner._load_history_cache = lambda _s: []
    runner._logger = type('L', (), {'warning': lambda *a, **k: None})()
    prev_bar = _bar(datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc), 100.0)
    incoming = _bar(datetime(2026, 1, 1, 9, 17, tzinfo=timezone.utc), 101.0)
    repaired = runner._repair_candle_gap('NFO:NIFTY26JAN25000CE', prev_bar, incoming)
    assert repaired == []


def test_gap_repair_generates_for_large_gap() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._main_loop = None
    runner._gap_repair_inflight = set()
    runner._load_history_cache = lambda _s: []
    runner._logger = type('L', (), {'warning': lambda *a, **k: None, 'info': lambda *a, **k: None})()
    prev_bar = _bar(datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc), 100.0)
    incoming = _bar(datetime(2026, 1, 1, 9, 20, tzinfo=timezone.utc), 101.0)
    repaired = runner._repair_candle_gap('NSE:NIFTY', prev_bar, incoming)
    assert len(repaired) == 4
    assert all(bar.synthetic for bar in repaired)


def test_gap_repair_does_not_create_synthetic_option_bars() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._main_loop = None
    runner._gap_repair_inflight = set()
    runner._load_history_cache = lambda _s: []
    runner._logger = type('L', (), {'warning': lambda *a, **k: None, 'info': lambda *a, **k: None})()
    runner._should_log_throttled = lambda *_a, **_k: False
    prev_bar = _bar(datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc), 100.0)
    incoming = _bar(datetime(2026, 1, 1, 9, 20, tzinfo=timezone.utc), 101.0)
    repaired = runner._repair_candle_gap('NFO:NIFTY26JAN25000CE', prev_bar, incoming)
    assert repaired == []


def test_backfill_skips_in_progress_partial_minute_and_dedupes() -> None:
    """Zerodha minute history includes the current PARTIAL candle. Backfill
    must skip it (the live builder owns the current minute) or the correct
    closed bar is later dropped as out-of-order and indicators keep the
    partial OHLC (observed all-day in the 2026-07-06 Railway logs)."""
    import logging
    from datetime import datetime, timedelta, timezone

    from nifty_scalper_bot.strategies.runner import OneMinuteBar, StrategyRunner

    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = logging.getLogger("test")
    r._last_bar_ts = {}
    r._symbol_history = {}
    r._should_log_throttled = lambda _k, _s: True

    now_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Monotonic gate: equal-ts duplicate and older bar are both dropped.
    r._last_bar_ts["X"] = now_min
    dup = OneMinuteBar(open=1, high=1, low=1, close=1, volume=0,
                       start=now_min, end=now_min + timedelta(minutes=1))
    r._ingest_bar("X", dup)
    older = OneMinuteBar(open=1, high=1, low=1, close=1, volume=0,
                         start=now_min - timedelta(minutes=2),
                         end=now_min - timedelta(minutes=1))
    r._ingest_bar("X", older)
    assert r._last_bar_ts["X"] == now_min

    # Backfill ingest: current (partial) minute skipped, closed minute kept.
    seen: list = []
    r._ingest_bar = lambda _s, bar, is_backfill=False: seen.append(bar.start)
    r._ingest_historical_bar_unlocked(
        {"symbol": "X", "timestamp": now_min, "open": 1, "high": 1,
         "low": 1, "close": 1, "volume": 0}
    )
    assert seen == [], "partial current-minute bar must not be ingested"
    closed_ts = now_min - timedelta(minutes=1)
    r._ingest_historical_bar_unlocked(
        {"symbol": "X", "timestamp": closed_ts, "open": 1, "high": 1,
         "low": 1, "close": 1, "volume": 0}
    )
    assert seen == [closed_ts]
