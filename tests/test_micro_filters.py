from types import SimpleNamespace

from src.execution.micro_filters import cap_for_mid, evaluate_micro


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        micro={
            "mode": "HARD",
            "max_spread_pct": 1.0,
            "depth_min_lots": 1,
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
    quote = {"bid": 155.0, "ask": 156.35, "bid_qty": 50, "ask_qty": 50}
    micro = evaluate_micro(quote, lot_size=50, atr_pct=0.03, cfg=cfg)
    assert micro["would_block"] is False
    shallow = {"bid": 155.0, "ask": 156.35, "bid_qty": 10, "ask_qty": 10}
    micro2 = evaluate_micro(shallow, lot_size=50, atr_pct=0.03, cfg=cfg)
    assert micro2["depth_ok"] is False
    assert micro2["would_block"] is True
