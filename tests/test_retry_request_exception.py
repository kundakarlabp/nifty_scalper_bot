import requests
import pytest

from src.data.source import _retry


def test_retry_handles_request_exception(monkeypatch):
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("boom")

    with pytest.raises(requests.exceptions.ConnectionError):
        _retry(boom, tries=3, base_delay=0)
    assert calls["n"] == 3
