from datetime import datetime, timedelta

import pytest

from src.data import source


def test_fetch_ohlc_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_download(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("boom")

    # Provide __version__ so backoff path is exercised
    fake_yf = type("T", (), {"download": fake_download, "__version__": "1"})
    monkeypatch.setattr(source, "yf", fake_yf)
    source._yf_last_fail_ts.clear()

    # Verify mark_fail/mark_success helpers
    source._yf_mark_fail("X")
    assert source._yf_should_skip("X") is True
    source._yf_mark_success("X")
    assert source._yf_should_skip("X") is False

    start = datetime(2024, 1, 1)
    end = start + timedelta(minutes=1)

    out1 = source._fetch_ohlc_yf("^NSEI", start, end, "minute")
    assert out1 == []
    assert len(calls) == 1

    out2 = source._fetch_ohlc_yf("^NSEI", start, end, "minute")
    assert out2 is None
    assert len(calls) == 1
