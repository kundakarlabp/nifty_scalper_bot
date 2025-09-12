from src.logs.journal import Journal


def test_journal_roundtrip(tmp_path):
    db = tmp_path / "journal.sqlite"
    j = Journal.open(str(db))
    j.append_event(
        ts="2024-01-01T00:00:00",
        trade_id="T1",
        leg_id="L1",
        etype="NEW",
        payload={"side": "BUY", "symbol": "NIFTY", "qty": 1, "limit_price": 100.0},
    )
    j.append_event(
        ts="2024-01-01T00:01:00",
        trade_id="T1",
        leg_id="L1",
        etype="PARTIAL",
        payload={"filled_qty": 1, "avg_price": 100.0},
    )
    legs = j.rehydrate_open_legs()
    assert len(legs) == 1
    trade = {
        "ts_entry": "2024-01-01T00:00:00",
        "ts_exit": "2024-01-01T00:02:00",
        "trade_id": "T1",
        "side": "BUY",
        "symbol": "NIFTY",
        "qty": 1,
        "entry": 100.0,
        "exit": 101.0,
        "exit_reason": "TP",
        "R": 1.0,
        "pnl_R": 1.0,
        "pnl_rupees": 75.0,
    }
    j.append_trade(trade)
    rows = j.last_trades(1)
    assert rows and rows[0]["trade_id"] == "T1"
