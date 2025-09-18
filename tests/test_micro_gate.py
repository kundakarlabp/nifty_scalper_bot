from src.config import settings
from src.execution.order_executor import micro_ok


def test_micro_ok_missing_bid_ask_returns_false_with_reason():
    quote = {"bid": 0, "ask": 0, "bid5_qty": 0, "ask5_qty": 0}
    ok, meta = micro_ok(
        quote,
        qty_lots=1,
        lot_size=50,
        max_spread_pct=0.35,
        depth_mult=5,
    )
    assert ok is False
    assert meta["block_reason"] == "no_quote"
    assert meta["spread_ok"] is True
    assert meta["depth_ok"] is False
    assert meta["depth_missing"] is True


def test_micro_ok_passes_with_valid_spread_and_depth():
    quote = {"bid": 100.0, "ask": 100.2, "bid5_qty": 1000, "ask5_qty": 1000}
    ok, meta = micro_ok(
        quote,
        qty_lots=1,
        lot_size=50,
        max_spread_pct=0.35,
        depth_mult=5,
    )
    assert ok is True
    assert meta is not None
    assert meta["depth_ok"] is True
    assert meta["spread_pct"] <= 0.35
    assert meta["required_qty"] == 250
    assert meta["depth_available"] >= 250
    assert meta["bid"] == 100.0
    assert meta["ask"] == 100.2


def test_micro_ok_allows_ltp_only_when_depth_missing():
    quote = {"bid": 0.0, "ask": 0.0, "bid5_qty": 0, "ask5_qty": 0, "ltp": 100.0}
    ok, meta = micro_ok(
        quote,
        qty_lots=1,
        lot_size=50,
        max_spread_pct=0.35,
        depth_mult=5,
    )
    assert ok is True
    assert meta is not None
    assert meta["depth_ok"] is True
    assert meta["spread_pct"] == getattr(settings.executor, "default_spread_pct_est", 0.25)
    assert meta["block_reason"] is None
