from datetime import datetime, timedelta
import time

from src.data import source


def test_fetch_ohlc_backoff(monkeypatch):
    """yfinance failures should trigger temporary backoff."""
    source._yf_last_fail_ts.clear()
    calls = []

    def boom(*_args, **_kwargs):
        calls.append(1)
        raise Exception("boom")

    fake_yf = type("T", (), {"download": boom, "__version__": "1.0"})
    monkeypatch.setattr(source, "yf", fake_yf)
    monkeypatch.setattr(source, "_warmup_next_try_ts", 0.0)

    start = datetime(2024, 1, 1)
    end = start + timedelta(minutes=1)
    try:
        assert source._fetch_ohlc_yf("FOO", start, end, "minute") is None
        assert source._fetch_ohlc_yf("FOO", start, end, "minute") is None
        assert len(calls) == 1
    finally:
        source._yf_last_fail_ts.clear()
        source._warmup_next_try_ts = 0.0


def test_fetch_ohlc_backoff_flag(monkeypatch):
    """Disabled warmup should set next try timestamp and increment backoff."""
    monkeypatch.setattr(source, "_warmup_next_try_ts", 0.0)
    monkeypatch.setattr(source, "_warmup_backoff", 1.0)
    monkeypatch.setattr(source, "DATA_WARMUP_DISABLE", True)
    monkeypatch.setattr(source, "YFINANCE_DISABLE", False)
    start = datetime(2024, 1, 1)
    end = start + timedelta(minutes=1)
    now = time.time()
    assert source._fetch_ohlc_yf("FOO", start, end, "minute") is None
    assert source._warmup_next_try_ts >= now + 1.0
    assert source._warmup_backoff == 2.0
