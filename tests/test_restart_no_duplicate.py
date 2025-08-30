import logging
from datetime import datetime

from src.logs.journal import Journal
from src.execution.order_executor import OrderExecutor, OrderReconciler


def test_restart_no_duplicate(tmp_path):
    db = tmp_path / "journal.sqlite"
    j = Journal.open(str(db))
    j.append_event(
        ts="2024-01-01T00:00:00",
        trade_id="T1",
        leg_id="T1:ENTRY",
        etype="NEW",
        payload={"side": "BUY", "symbol": "NIFTY", "qty": 1, "limit_price": 100.0},
    )
    j.append_event(
        ts="2024-01-01T00:00:01",
        trade_id="T1",
        leg_id="T1:ENTRY",
        etype="ACK",
        broker_order_id="OID1",
        idempotency_key="ID1",
        payload={"price": 100.0},
    )
    j2 = Journal.open(str(db))
    legs = j2.rehydrate_open_legs()

    class MockKite:
        def __init__(self) -> None:
            self.place_order_called = False

        def place_order(self, **payload):
            self.place_order_called = True

        def orders(self):
            return [
                {
                    "order_id": "OID1",
                    "status": "OPEN",
                    "filled_quantity": 0,
                    "average_price": 0.0,
                    "tag": "ID1",
                }
            ]

    kite = MockKite()
    executor = OrderExecutor(kite=kite, telegram_controller=None, journal=j2)
    fsm = executor.get_or_create_fsm("T1")
    executor.attach_leg_from_journal(fsm, legs[0])
    reconciler = OrderReconciler(kite, executor, logging.getLogger("test"))
    reconciler.step(datetime.utcnow())
    executor.step_queue()
    assert not kite.place_order_called
