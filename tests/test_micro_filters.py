from types import SimpleNamespace

from src.execution.micro_filters import cap_for_mid, evaluate_micro


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        micro={
            "mode": "HARD",
            "max_spread_pct": 1.0,
            "depth_min_lots": 1,
            "depth_multiplier": 5,
            "require_depth": True,
            "dynamic": True,
            "table": [
                {"min_mid": 0, "cap_pct": 2.0},
                {"min_mid": 50, "cap_pct": 1.5},
                {"min_mid": 100, "cap_pct": 1.2},
                {"min_mid": 150, "cap_pct": 1.0},
            ],
        }
    )


def test_cap_for_mid_respects_table():
    cfg = _cfg()
    assert cap_for_mid(120, cfg) == 1.2
    assert cap_for_mid(160, cfg) == 1.0


def test_evaluate_micro_flags_depth():
    cfg = _cfg()
    depth_ok_quote = {
        "bid": 155.0,
        "ask": 156.35,
        "bid5_qty": [120, 110, 100, 90, 80],
        "ask5_qty": [120, 110, 100, 90, 80],
    }
    micro = evaluate_micro(
        depth_ok_quote, lot_size=50, atr_pct=0.03, cfg=cfg, side="BUY"
    )
    assert micro["would_block"] is False
    assert micro["depth_ok"] is True
    assert micro["depth_available"] >= micro["required_qty"]
    shallow = {
        "bid": 155.0,
        "ask": 156.35,
        "bid5_qty": [40, 40, 40, 40, 40],
        "ask5_qty": [40, 40, 40, 40, 40],
    }
    micro2 = evaluate_micro(shallow, lot_size=50, atr_pct=0.03, cfg=cfg, side="BUY")
    assert micro2["depth_ok"] is False
    assert micro2["would_block"] is True


def test_evaluate_micro_uses_side_for_depth():
    cfg = _cfg()
    quote = {
        "bid": 155.0,
        "ask": 156.0,
        "bid5_qty": [20, 15, 10, 5, 5],
        "ask5_qty": [120, 110, 100, 90, 80],
    }
    buy = evaluate_micro(quote, lot_size=50, atr_pct=0.03, cfg=cfg, side="BUY")
    sell = evaluate_micro(quote, lot_size=50, atr_pct=0.03, cfg=cfg, side="SELL")
    assert buy["depth_ok"] is True
    assert sell["depth_ok"] is False
    assert sell["would_block"] is True


def test_evaluate_micro_requires_top5_arrays():
    cfg = _cfg()
    quote = {"bid": 155.0, "ask": 156.35}
    result = evaluate_micro(quote, lot_size=50, atr_pct=0.03, cfg=cfg, side="BUY")
    assert result["reason"] == "no_quote"
    assert result["would_block"] is True
