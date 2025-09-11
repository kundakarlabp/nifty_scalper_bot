from src.execution.order_executor import OrderExecutor
from src.state.store import StateStore

def test_restart_resumes_trailing(tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    ex1 = OrderExecutor(kite=None, state_store=store)
    payload = {
        "action": "BUY",
        "symbol": "TEST",
        "quantity": ex1.lot_size,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "trail_atr_mult": 1.0,
        "client_oid": "RID1",
    }
    rid = ex1.place_order(payload)
    store.record_order("RID1", payload)
    ex1.update_trailing_stop(rid, current_price=101.0, atr=1.0)
    sl_before = ex1._active[rid].sl_price

    ex2 = OrderExecutor(kite=None, state_store=store)
    snap = store.snapshot()
    ex2.restore_record("RID1", snap.open_orders["RID1"])
    ex2.update_trailing_stop("RID1", current_price=102.0, atr=1.0)
    assert ex2._active["RID1"].sl_price > sl_before
