from types import SimpleNamespace
from src.risk.position_sizing import lots_from_premium_cap, PositionSizer
from src.config import settings as cfg


def test_lots_from_premium_cap_equity_sufficient(monkeypatch):
    """Returns at least one lot when equity-based cap allows it."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 0.0, raising=False)
    runner = SimpleNamespace(equity_amount=40_000.0)
    price = 112.8
    lot_size = 75
    lots, unit, cap = lots_from_premium_cap(
        runner, {"mid": price}, lot_size=lot_size, max_lots=5
    )
    assert lots == 1
    assert unit == price * lot_size
    assert cap == 40_000.0 * 0.40


def test_lots_from_premium_cap_equity_insufficient(monkeypatch):
    """Blocks when cap derived from equity is below one lot."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 0.0, raising=False)
    runner = SimpleNamespace(equity_amount=20_000.0)
    price = 112.8
    lot_size = 75
    lots, unit, cap = lots_from_premium_cap(
        runner, {"mid": price}, lot_size=lot_size, max_lots=5
    )
    assert lots == 0
    assert cap == 20_000.0 * 0.40
    assert cap < unit
    sizer = PositionSizer()
    qty, sized_lots, diag = sizer.size_from_signal(
        entry_price=price,
        stop_loss=90.0,
        lot_size=lot_size,
        equity=20_000.0,
        spot_sl_points=5.0,
        delta=0.5,
    )
    assert qty == 0 and sized_lots == 0
    assert diag["block_reason"] == "cap_lt_one_lot"
    assert diag["cap_abs"] is None


def test_lots_from_premium_cap_fallbacks_to_default_equity(monkeypatch):
    """When live equity is unavailable, the default equity should be used."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.50, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 0.0, raising=False)
    monkeypatch.setattr(cfg.risk, "default_equity", 40_000.0, raising=False)
    monkeypatch.setattr(cfg.risk, "use_live_equity", False, raising=False)
    runner = SimpleNamespace()
    lots, _, cap = lots_from_premium_cap(runner, {"mid": 100.0}, lot_size=50, max_lots=5)
    assert cap == 40_000.0 * 0.50
    assert lots == 4


def test_lots_from_premium_cap_respects_absolute_cap(monkeypatch):
    """Absolute premium caps should clamp exposure even if equity allows more."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.80, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 5_000.0, raising=False)
    runner = SimpleNamespace(equity_amount=100_000.0)
    lots, unit, cap = lots_from_premium_cap(
        runner,
        {"mid": 100.0},
        lot_size=50,
        max_lots=10,
    )
    assert cap == 5_000.0
    assert unit == 5_000.0
    assert lots == 1

