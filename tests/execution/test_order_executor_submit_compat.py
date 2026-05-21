import pytest

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.order_executor import OrderExecutor


class _MDM:
    def now_ns(self):
        return 1_000_000_000

    def get_last_quote(self, symbol):
        return {
            "bid": 100,
            "ask": 101,
            "ts_ns": self.now_ns(),
        }


class _KwargsBroker:
    def __init__(self):
        self.calls = 0
        self.received = None

    def place_order(self, **kwargs):
        self.calls += 1
        self.received = kwargs
        return {"order_id": "KWARGS_ORDER"}


class _PayloadBroker:
    def __init__(self):
        self.calls = 0
        self.received = None

    def place_order(self, payload):
        self.calls += 1
        self.received = payload
        return {"order_id": "PAYLOAD_ORDER"}


class _InternalTypeErrorBroker:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        raise TypeError("unsupported operand type after broker response")


class _RejectBroker:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        raise ValueError("broker rejected order")


def _executor(broker):
    risk = RiskConfig(
        max_daily_trades=100,
        max_order_notional=10_000_000,
        allow_short=True,
    )
    return OrderExecutor(broker, risk, _MDM())


def test_order_executor_submit_supports_kwargs_broker():
    broker = _KwargsBroker()
    ex = _executor(broker)

    result = ex._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})

    assert result["order_id"] == "KWARGS_ORDER"
    assert broker.calls == 1
    assert broker.received == {"symbol": "NFO:NIFTY", "quantity": 75}


def test_order_executor_submit_supports_payload_broker_signature_fallback():
    broker = _PayloadBroker()
    ex = _executor(broker)

    result = ex._submit_broker_order(
        {"symbol": "NFO:NIFTY", "quantity": 75},
        legacy_payload={"symbol": "NFO:NIFTY", "qty": 75, "type": "LIMIT"},
    )

    assert result["order_id"] == "PAYLOAD_ORDER"
    assert broker.calls == 1
    assert broker.received == {"symbol": "NFO:NIFTY", "qty": 75, "type": "LIMIT"}


def test_order_executor_submit_does_not_fallback_on_internal_type_error():
    broker = _InternalTypeErrorBroker()
    ex = _executor(broker)

    with pytest.raises(TypeError, match="unsupported operand type"):
        ex._submit_broker_order(
            {"symbol": "NFO:NIFTY", "quantity": 75},
            legacy_payload={"symbol": "NFO:NIFTY", "qty": 75},
        )

    assert broker.calls == 1


def test_order_executor_submit_does_not_fallback_on_non_typeerror_rejection():
    broker = _RejectBroker()
    ex = _executor(broker)

    with pytest.raises(ValueError, match="broker rejected"):
        ex._submit_broker_order(
            {"symbol": "NFO:NIFTY", "quantity": 75},
            legacy_payload={"symbol": "NFO:NIFTY", "qty": 75},
        )

    assert broker.calls == 1
