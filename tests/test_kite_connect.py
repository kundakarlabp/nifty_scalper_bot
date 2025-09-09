from __future__ import annotations

import pytest

from src.broker.interface import BrokerError
from src.brokers import kite
from src.brokers.kite import KiteBroker


def test_connect_translates_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyKiteConnect:
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - simple stub
            raise Exception("boom")

    monkeypatch.setattr(kite, "KiteConnect", DummyKiteConnect)
    broker = KiteBroker(api_key="k", access_token="t", enable_ws=False)
    with pytest.raises(BrokerError) as err:
        broker.connect()
    assert "boom" in str(err.value)
