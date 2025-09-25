from __future__ import annotations

from datetime import datetime, timedelta

from src.logs.journal import Journal, read_trades_between


def test_journal_event_roundtrip(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = Journal.open(str(path))

    journal.append_event(
        ts="2024-01-01T09:15:00",
        trade_id="T1",
        leg_id="L1",
        etype="NEW",
        payload={"side": "BUY", "symbol": "NIFTY", "qty": 50, "limit_price": 120.5},
    )
    journal.append_event(
        ts="2024-01-01T09:16:00",
        trade_id="T1",
        leg_id="L1",
        etype="OPEN",
        broker_order_id="BO1",
        idempotency_key="IK1",
        payload={"filled_qty": 10, "avg_price": 121.0},
    )

    assert journal.get_idemp_leg("IK1") == "L1"

    open_legs = journal.rehydrate_open_legs()
    assert open_legs == [
        {
            "trade_id": "T1",
            "leg_id": "L1",
            "state": "OPEN",
            "side": "BUY",
            "symbol": "NIFTY",
            "qty": 50,
            "limit_price": 120.5,
            "filled_qty": 10,
            "avg_price": 121.0,
            "broker_order_id": "BO1",
            "idempotency_key": "IK1",
        }
    ]

    journal._conn.close()


def test_journal_checkpoint_and_trades(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = Journal.open(str(path))

    journal.append_trade(
        {
            "ts_entry": "2024-01-01T09:15:00",
            "ts_exit": "2024-01-01T09:45:00",
            "trade_id": "T1",
            "side": "BUY",
            "symbol": "NIFTY",
            "qty": 50,
            "entry": 120.0,
            "exit": 121.0,
            "exit_reason": "TARGET",
            "R": 1.2,
            "pnl_R": 1.2,
            "pnl_rupees": 1000.0,
        }
    )

    journal.save_checkpoint({"foo": "bar", "ts": datetime(2024, 1, 1, 10, 0)})
    checkpoint = journal.load_latest_checkpoint()
    assert checkpoint == {"foo": "bar", "ts": "2024-01-01T10:00:00"}

    start = datetime.fromisoformat("2024-01-01T09:30:00") - timedelta(minutes=5)
    end = datetime.fromisoformat("2024-01-01T10:00:00")
    trades = read_trades_between(start, end, path=str(path))
    assert trades == [{"ts_close": datetime.fromisoformat("2024-01-01T09:45:00"), "pnl_R": 1.2}]

    journal._conn.close()
