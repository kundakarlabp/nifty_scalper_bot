from src.execution.order_executor import OrderExecutor


def test_setup_gtt_orders_emits_postfill_event() -> None:
    exe = OrderExecutor(kite=None)
    exe.set_partial_enabled(True)
    payload = {
        "action": "BUY",
        "quantity": 150,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "symbol": "TEST-CE",
        "client_oid": "oid-123",
        "trace_id": "trace-xyz",
        "sizing": {"risk": 1},
    }
    record_id = exe.place_order(payload)
    assert record_id is not None

    # Drain entry events so we only validate postfill payload.
    exe.drain_order_events()

    exe.setup_gtt_orders(record_id, sl_price=94.0, tp_price=112.0)

    events = exe.drain_order_events()
    postfill = next(e for e in events if e["event"] == "postfill_setup")

    assert postfill["trace_id"] == "trace-xyz"
    assert postfill["client_oid"] == "oid-123"
    assert postfill["mode"] == "paper"
    assert postfill["sizing"] == {"risk": 1}
    assert postfill["qty"] == 150
    assert postfill["tp_order_ids"] == [None, None]

    tp_meta = postfill["tp_meta"]
    assert tp_meta["tp1"]["qty"] == 50
    assert tp_meta["tp1"]["attempts"] == 1
    assert tp_meta["tp1"]["retries"] == 0
    assert tp_meta["tp1"]["retry_delays_ms"] is None

    assert tp_meta["tp2"]["qty"] == 100
    assert tp_meta["tp2"]["attempts"] == 1
    assert tp_meta["tp2"]["retries"] == 0
    assert tp_meta["tp2"]["retry_delays_ms"] is None
