from src.execution.order_executor import OrderExecutor, _OrderRecord


def test_eod_flatten_cancels_and_exits():
    class Kite:
        def __init__(self):
            self.cancelled = []
            self.gtt_cancelled = []
            self.placed = []

        def cancel_order(self, variety, order_id):
            self.cancelled.append(order_id)

        def cancel_gtt(self, gtt_id):
            self.gtt_cancelled.append(gtt_id)

        def positions(self):
            return {"day": [{"tradingsymbol": "TEST", "quantity": 1}]}

        def place_order(self, **req):
            self.placed.append(req)
            return {"order_id": "exit"}

    kite = Kite()
    ex = OrderExecutor(kite=kite)
    rec = _OrderRecord(
        order_id="RID1",
        instrument_token=1,
        symbol="TEST",
        side="BUY",
        quantity=ex.lot_size,
        entry_price=100.0,
        tick_size=0.05,
        sl_gtt_id=10,
        child_order_ids=["CID1"],
    )
    ex._active[rec.record_id] = rec

    ex.cancel_all_orders()
    assert kite.cancelled == ["CID1"]
    assert kite.gtt_cancelled == [10]

    ex.close_all_positions_eod()
    assert kite.placed and kite.placed[0]["transaction_type"] == "SELL"
