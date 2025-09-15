from src.risk.position_sizing import lots_from_premium_cap, PositionSizer
from src.config import settings as cfg


def test_lots_from_premium_cap_equity_sufficient(monkeypatch):
    """Returns at least one lot when equity-based cap allows it."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.25, raising=False)
    lots, unit, cap = lots_from_premium_cap(
        None, {"mid": 200.0}, lot_size=25, max_lots=5, equity=30_000.0
    )
    assert lots == 1
    assert unit == 200.0 * 25
    assert cap == 30_000.0 * 0.25


def test_lots_from_premium_cap_equity_insufficient(monkeypatch):
    """Blocks when cap derived from equity is below one lot."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.25, raising=False)
    lots, unit, cap = lots_from_premium_cap(
        None, {"mid": 200.0}, lot_size=25, max_lots=5, equity=10_000.0
    )
    assert lots == 0
    assert cap == 10_000.0 * 0.25
    assert cap < unit
    sizer = PositionSizer()
    qty, sized_lots, diag = sizer.size_from_signal(
        entry_price=200.0,
        stop_loss=180.0,
        lot_size=25,
        equity=10_000.0,
        spot_sl_points=20.0,
        delta=0.5,
    )
    assert qty == 0 and sized_lots == 0
    assert diag["block_reason"] == "cap_lt_one_lot"

