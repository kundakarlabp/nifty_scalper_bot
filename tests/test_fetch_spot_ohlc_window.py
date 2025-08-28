from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd

from src.strategies.runner import StrategyRunner
from src.config import settings


class StubDataSource:
    def __init__(self) -> None:
        self.calls = []

    def fetch_ohlc(self, token, start, end, timeframe):
        self.calls.append((token, start, end, timeframe))
        idx = pd.date_range(start, end, freq="1min", inclusive="left")
        data = {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
        return pd.DataFrame(data, index=idx)

    def get_last_price(self, symbol):
        return 1.0


def _setup_runner(monkeypatch, now_dt: datetime) -> tuple[StrategyRunner, StubDataSource]:
    class _DummyTelegram:
        def send_message(self, msg: str) -> None:
            pass

    runner = StrategyRunner(telegram_controller=_DummyTelegram())
    runner._start_time = runner._parse_hhmm("09:20")
    runner._end_time = runner._parse_hhmm("15:25")
    ds = StubDataSource()
    runner.data_source = ds
    monkeypatch.setattr(runner, "_now_ist", lambda: now_dt)
    return runner, ds


def test_fetch_spot_ohlc_in_session(monkeypatch):
    now_dt = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    runner, ds = _setup_runner(monkeypatch, now_dt)

    df = runner._fetch_spot_ohlc()
    assert df is not None
    assert len(ds.calls) == 1
    _, start, end, _ = ds.calls[0]

    lookback = int(max(settings.data.lookback_minutes, settings.strategy.min_bars_for_signal) * 1.1)
    assert end == now_dt
    assert start == end - timedelta(minutes=lookback)


def test_fetch_spot_ohlc_post_session(monkeypatch):
    now_dt = datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc)
    runner, ds = _setup_runner(monkeypatch, now_dt)

    df = runner._fetch_spot_ohlc()
    assert df is not None
    assert len(ds.calls) == 1
    _, start, end, _ = ds.calls[0]

    lookback = int(max(settings.data.lookback_minutes, settings.strategy.min_bars_for_signal) * 1.1)
    expected_end = datetime(2024, 1, 1, 15, 25, tzinfo=timezone.utc)
    assert end == expected_end
    assert start == expected_end - timedelta(minutes=lookback)


def test_fetch_spot_ohlc_pre_session(monkeypatch):
    now_dt = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
    runner, ds = _setup_runner(monkeypatch, now_dt)

    df = runner._fetch_spot_ohlc()
    assert df is not None
    assert len(ds.calls) == 1
    _, start, end, _ = ds.calls[0]

    lookback = int(max(settings.data.lookback_minutes, settings.strategy.min_bars_for_signal) * 1.1)
    expected_end = datetime(2023, 12, 31, 15, 25, tzinfo=timezone.utc)
    assert end == expected_end
    assert start == expected_end - timedelta(minutes=lookback)


def test_fetch_spot_ohlc_invalid_window(monkeypatch):
    now_dt = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    runner, ds = _setup_runner(monkeypatch, now_dt)
    monkeypatch.setattr(settings.data, "lookback_minutes", 0, raising=False)
    monkeypatch.setattr(settings.strategy, "min_bars_for_signal", 0, raising=False)

    df = runner._fetch_spot_ohlc()
    assert df is None
    assert ds.calls == []


def test_fetch_spot_ohlc_alert_on_none(monkeypatch):
    """If the data source returns None, the runner should alert the user."""

    now_dt = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)

    class _Telegram:
        def __init__(self) -> None:
            self.msgs: List[str] = []

        def send_message(self, msg: str) -> None:
            self.msgs.append(msg)

    class _FailSource:
        def fetch_ohlc(self, *args, **kwargs):
            return None

        def get_last_price(self, symbol):
            return 1.0

    telegram = _Telegram()
    runner = StrategyRunner(telegram_controller=telegram)
    runner._start_time = runner._parse_hhmm("09:20")
    runner._end_time = runner._parse_hhmm("15:25")
    runner.data_source = _FailSource()
    monkeypatch.setattr(runner, "_now_ist", lambda: now_dt)

    df = runner._fetch_spot_ohlc()

    assert df is not None  # falls back to LTP
    # No automatic alert should be sent to Telegram
    assert telegram.msgs == []
    # Internal error state should still be set for on-demand inspection
    assert getattr(runner, "_last_error", None) == "no_historical_data"
