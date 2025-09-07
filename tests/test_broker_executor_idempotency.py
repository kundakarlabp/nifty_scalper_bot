from __future__ import annotations

from src.execution.broker_executor import BrokerOrderExecutor


class DummyBroker:
    """Minimal broker stub recording order submissions."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_req = None

    def place_order(self, req):
        self.calls += 1
        self.last_req = req
        return f"OID{self.calls}"


def test_dedup_and_generation() -> None:
    broker = DummyBroker()
    exe = BrokerOrderExecutor(broker, instrument_id_mapper=lambda s: 1)

    oid1 = exe.buy("SYM", 1)
    cid = broker.last_req.client_order_id
    assert cid is not None
    assert broker.calls == 1

    oid2 = exe.buy("SYM", 1, client_order_id=cid)
    assert broker.calls == 1
    assert oid2 == oid1
