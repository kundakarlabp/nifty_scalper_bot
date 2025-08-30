from datetime import datetime

from src.logs.journal import Journal
from src.execution.order_executor import OrderExecutor


def test_rehydrate_fsm(tmp_path):
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
    j.append_event(
        ts="2024-01-01T00:00:02",
        trade_id="T1",
        leg_id="T1:ENTRY",
        etype="PARTIAL",
        broker_order_id="OID1",
        payload={"filled_qty": 1, "avg_price": 100.0},
    )
    legs = j.rehydrate_open_legs()
    oe = OrderExecutor(kite=None, telegram_controller=None, journal=j)
    fsm = oe.get_or_create_fsm("T1")
    oe.attach_leg_from_journal(fsm, legs[0])
    leg = fsm.legs["T1:ENTRY"]
    assert leg.state.name == "PARTIAL"
    assert leg.filled_qty == 1
    assert oe._idemp_map["ID1"] == leg.leg_id
