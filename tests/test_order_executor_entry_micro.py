from src.execution import order_executor as oe


def test_place_order_blocks_on_micro(monkeypatch):
    monkeypatch.setattr(oe, "ENTRY_WAIT_S", 0)
    ex = oe.OrderExecutor(None)
    payload = {
        "action": "BUY",
        "quantity": ex.lot_size,
        "entry_price": 100.0,
        "bid": 100.0,
        "ask": 101.0,
        "depth": 0.0,
        "symbol": "TEST",
    }
    oid = ex.place_order(payload)
    assert oid is None
    assert ex.last_error == "micro_block"
