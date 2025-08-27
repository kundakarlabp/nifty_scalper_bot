import pandas as pd
from datetime import datetime, timezone, timedelta
from src.strategies.runner import StrategyRunner
from src.config import settings

class DummyTelegram:
    def send_message(self, msg: str) -> None:
        pass

class DummySource:
    def __init__(self) -> None:
        self.calls = []
    def fetch_ohlc(self, token, start, end, timeframe):
        self.calls.append((token, start, end, timeframe))
        return pd.DataFrame({"open":[1],"high":[1],"low":[1],"close":[1],"volume":[1]}, index=[end])
    def get_last_price(self, sym):
        return 100
    def connect(self):
        pass

IST = timezone(timedelta(hours=5, minutes=30))

def make_runner(monkeypatch, start_str="09:15", end_str="15:30", lookback=30):
    monkeypatch.setattr(settings.data, "time_filter_start", start_str, raising=False)
    monkeypatch.setattr(settings.data, "time_filter_end", end_str, raising=False)
    monkeypatch.setattr(settings.data, "lookback_minutes", lookback, raising=False)
    monkeypatch.setattr(settings.strategy, "min_bars_for_signal", 1, raising=False)
    monkeypatch.setattr("src.strategies.runner.LiveKiteSource", None)
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.data_source = DummySource()
    monkeypatch.setattr(settings.instruments, "instrument_token", 111, raising=False)
    return runner

def test_fetch_within_session(monkeypatch):
    runner = make_runner(monkeypatch)
    now = datetime(2024,1,1,10,0,tzinfo=IST)
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._fetch_spot_ohlc()
    _, start, end, _ = runner.data_source.calls[-1]
    assert start == datetime(2024,1,1,9,30,tzinfo=IST)
    assert end == datetime(2024,1,1,10,0,tzinfo=IST)

def test_fetch_after_session(monkeypatch):
    runner = make_runner(monkeypatch)
    now = datetime(2024,1,1,16,0,tzinfo=IST)
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._fetch_spot_ohlc()
    _, start, end, _ = runner.data_source.calls[-1]
    assert start == datetime(2024,1,1,15,0,tzinfo=IST)
    assert end == datetime(2024,1,1,15,30,tzinfo=IST)

def test_fetch_before_session(monkeypatch):
    runner = make_runner(monkeypatch)
    now = datetime(2024,1,2,8,0,tzinfo=IST)
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner._fetch_spot_ohlc()
    _, start, end, _ = runner.data_source.calls[-1]
    assert start == datetime(2024,1,1,15,0,tzinfo=IST)
    assert end == datetime(2024,1,1,15,30,tzinfo=IST)

def test_invalid_window_returns_none(monkeypatch):
    runner = make_runner(monkeypatch, start_str="15:30", end_str="15:00")
    now = datetime(2024,1,1,16,0,tzinfo=IST)
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    result = runner._fetch_spot_ohlc()
    assert result is None
    assert runner.data_source.calls == []
