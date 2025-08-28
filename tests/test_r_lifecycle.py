from src.execution.order_executor import OrderExecutor
from src.config import settings


def test_r_lifecycle_trailing(monkeypatch):
    monkeypatch.setattr(settings.executor, "partial_tp_enable", True, raising=False)
    monkeypatch.setattr(settings.executor, "tp1_qty_ratio", 0.5, raising=False)
    monkeypatch.setattr(settings.executor, "enable_trailing", True, raising=False)

    ex = OrderExecutor(kite=None)
    payload = {
        "action": "BUY",
        "quantity": ex.lot_size * 2,
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit": 102.0,
        "trail_atr_mult": 0.8,
        "symbol": "TEST",
    }
    rid = ex.place_order(payload)
    assert rid is not None

    ex.handle_tp1_fill(rid)
    rec = ex._active[rid]
    assert rec.quantity == ex.lot_size
    assert abs(rec.sl_price - 100.1) < 1e-6
    assert rec.trailing_mult == 0.8

    ex.update_trailing_stop(rid, current_price=101.5, atr=0.5)
    assert rec.sl_price >= 100.1
