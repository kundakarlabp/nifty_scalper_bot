from datetime import datetime, timedelta

from src.data.source import LiveKiteSource


def test_prev_session_close_fallback(monkeypatch) -> None:
    class DummyKite:
        def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
            return [
                {
                    "date": datetime(2024, 1, 1, 15, 30),
                    "open": 99.0,
                    "high": 99.0,
                    "low": 99.0,
                    "close": 99.0,
                    "volume": 0,
                }
            ]

    kite = DummyKite()
    src = LiveKiteSource(kite)
    monkeypatch.setattr(src.cb_hist, "allow", lambda: False)
    monkeypatch.setattr(src, "get_last_price", lambda token: None)
    end = datetime(2024, 1, 2, 15, 30)
    start = end - timedelta(minutes=1)
    df = src._fetch_ohlc_df(token=123, start=start, end=end, timeframe="minute")
    assert not df.empty
    assert (df["close"] == 99.0).all()
    assert src._last_hist_reason == "synthetic_prev_close"


def test_prev_session_close_no_kite() -> None:
    src = LiveKiteSource(None)
    assert src._prev_session_close(1, "minute") is None


def test_prev_close_fallback_missing_prev_close(monkeypatch) -> None:
    class FailingKite:
        def historical_data(self, *args, **kwargs):
            raise Exception("boom")

    kite = FailingKite()
    src = LiveKiteSource(kite)
    monkeypatch.setattr(src.cb_hist, "allow", lambda: False)
    monkeypatch.setattr(src, "get_last_price", lambda token: None)
    end = datetime(2024, 1, 2, 15, 30)
    start = end - timedelta(minutes=1)
    df = src._fetch_ohlc_df(token=123, start=start, end=end, timeframe="minute")
    assert df.empty
    assert src._last_hist_reason == "broker_unavailable"


def test_prev_session_close_empty_rows(monkeypatch) -> None:
    class EmptyKite:
        def historical_data(self, *args, **kwargs):
            return []

    kite = EmptyKite()
    src = LiveKiteSource(kite)
    assert src._prev_session_close(1, "minute") is None


def test_prev_session_close_missing_close(monkeypatch) -> None:
    class NoCloseKite:
        def historical_data(self, *args, **kwargs):
            return [{"date": datetime(2024, 1, 1, 15, 30), "open": 1.0}]

    kite = NoCloseKite()
    src = LiveKiteSource(kite)
    assert src._prev_session_close(1, "minute") is None

