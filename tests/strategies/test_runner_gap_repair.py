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
    runner._logger = type('L', (), {'warning': lambda *a, **k: None})()
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
