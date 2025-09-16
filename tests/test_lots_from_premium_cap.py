from types import SimpleNamespace
from src.risk.position_sizing import lots_from_premium_cap, PositionSizer
from src.config import settings as cfg


def test_lots_from_premium_cap_equity_sufficient(monkeypatch):
    """Returns at least one lot when equity-based cap allows it."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    runner = SimpleNamespace(equity_amount=40_000.0)
    lots, unit, cap = lots_from_premium_cap(
        runner, {"mid": 112.8}, lot_size=75, max_lots=5
    )
    assert lots == 1
    assert unit == 112.8 * 75
    assert cap == 40_000.0 * 0.40


def test_lots_from_premium_cap_equity_insufficient(monkeypatch):
    """Blocks when cap derived from equity is below one lot."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    runner = SimpleNamespace(equity_amount=20_000.0)
    lots, unit, cap = lots_from_premium_cap(
        runner, {"mid": 112.8}, lot_size=75, max_lots=5
    )
    assert lots == 0
    assert cap == 20_000.0 * 0.40
    assert cap < unit
    sizer = PositionSizer()
    qty, sized_lots, diag = sizer.size_from_signal(
        entry_price=112.8,
        stop_loss=92.8,
        lot_size=75,
        equity=20_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 0 and sized_lots == 0
    assert diag["block_reason"] == "cap_lt_one_lot"

