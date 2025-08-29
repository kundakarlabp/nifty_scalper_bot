from src.execution.order_executor import micro_ok


def test_micro_ok_missing_bid_ask_returns_false_none():
    quote = {"bid": 0, "ask": 0, "bid5_qty": 0, "ask5_qty": 0}
    ok, meta = micro_ok(quote, qty_lots=1, lot_size=50, max_spread_pct=0.35, depth_mult=5)
    assert ok is False
    assert meta is None


def test_micro_ok_passes_with_valid_spread_and_depth():
    quote = {"bid": 100.0, "ask": 100.2, "bid5_qty": 1000, "ask5_qty": 1000}
    ok, meta = micro_ok(quote, qty_lots=1, lot_size=50, max_spread_pct=0.35, depth_mult=5)
    assert ok is True
    assert meta is not None
    assert meta["depth_ok"] is True
    assert meta["spread_pct"] <= 0.35
