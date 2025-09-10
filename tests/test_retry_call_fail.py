import pytest
from src.execution import order_executor as oe


def test_retry_call_propagates_network_exception(caplog):
    calls = {"count": 0}

    def failing():
        calls["count"] += 1
        raise oe.NetworkException("boom")

    with caplog.at_level("WARNING"):
        with pytest.raises(oe.NetworkException):
            oe._retry_call(failing, tries=2)
    assert "Transient broker error" in caplog.text
    assert calls["count"] == 2
