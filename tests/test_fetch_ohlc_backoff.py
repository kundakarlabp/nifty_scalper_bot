from datetime import datetime, timedelta

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

    start = datetime(2024, 1, 1)
    end = start + timedelta(minutes=1)
    try:
        assert source._fetch_ohlc_yf("FOO", start, end, "minute") is None
        assert source._fetch_ohlc_yf("FOO", start, end, "minute") is None
        assert len(calls) == 1
    finally:
        source._yf_last_fail_ts.clear()
