from types import SimpleNamespace
from types import SimpleNamespace

from src.risk.position_sizing import lots_from_premium_cap, PositionSizer
from src.config import settings as cfg


def test_lots_from_premium_cap_equity_sufficient(monkeypatch):
    """Returns at least one lot when equity-based cap allows it."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT", 40.0, raising=False)
    price = 112.8
    lot_size = 75
    lots, meta = lots_from_premium_cap(price, lot_size, cfg, 40_000.0)
    assert lots == 1
    assert meta["unit_notional"] == price * lot_size
    assert meta["cap"] == 40_000.0 * 0.40


def test_lots_from_premium_cap_equity_insufficient(monkeypatch):
    """Blocks when cap derived from equity is below one lot."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT", 4.0, raising=False)
    price = 112.8
    lot_size = 75
    lots, meta = lots_from_premium_cap(price, lot_size, cfg, 20_000.0)
    assert lots == 0
    assert meta["cap"] == 20_000.0 * 0.04
    assert meta["cap"] < meta["unit_notional"]
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


def test_lots_from_premium_cap_ignores_live_equity_when_disabled(monkeypatch):
    """Defaults to configured equity when live equity usage is disabled."""
    monkeypatch.setattr(cfg, "RISK_USE_LIVE_EQUITY", False, raising=False)
    monkeypatch.setattr(cfg, "RISK_DEFAULT_EQUITY", 60_000.0, raising=False)
    monkeypatch.setattr(cfg.risk, "use_live_equity", False, raising=False)
    monkeypatch.setattr(cfg.risk, "default_equity", 60_000.0, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT", 40.0, raising=False)
    price = 100.0
    lot_size = 50
    lots, meta = lots_from_premium_cap(price, lot_size, cfg, 5_000.0)
    assert lots >= 1
    assert meta["equity"] == 60_000.0
    assert meta["use_live_equity"] is False

