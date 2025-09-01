import pytest
from execution import broker_retry


def test_broker_retry_propagates_last_exception(monkeypatch) -> None:
    monkeypatch.setattr(broker_retry.time, "sleep", lambda _s: None)

    def boom() -> None:
        raise Exception("429 too many requests")

    with pytest.raises(Exception) as exc_info:
        broker_retry.call(boom)
    assert "429" in str(exc_info.value)
