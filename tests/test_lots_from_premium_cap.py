from src.risk.position_sizing import lots_from_premium_cap, PositionSizer
from src.config import settings as cfg


def test_lots_from_premium_cap_equity_sufficient(monkeypatch):
    """Returns at least one lot when equity-based cap allows it."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    price = 112.8
    lot_size = 75
    lots, meta = lots_from_premium_cap(
        premium=price,
        lot_size=lot_size,
        settings_obj=cfg,
        live_equity=40_000.0,
    )
    assert lots == 1
    assert meta["cap"] == 16_000.0
    assert meta["unit_notional"] == round(price * lot_size, 2)
    assert meta["cap_pct"] == 40.0
    assert meta["source"] == "equity"
    assert meta["equity_source"] == "live"
    assert meta["cap_abs"] is None
    assert meta.get("reason") is None


def test_lots_from_premium_cap_equity_insufficient(monkeypatch):
    """Blocks when cap derived from equity is below one lot."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.40, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    price = 112.8
    lot_size = 75
    lots, meta = lots_from_premium_cap(
        premium=price,
        lot_size=lot_size,
        settings_obj=cfg,
        live_equity=20_000.0,
    )
    assert lots == 0
    assert meta["cap"] == 8_000.0
    assert meta["unit_notional"] == round(price * lot_size, 2)
    assert meta["reason"] == "cap_lt_one_lot"
    assert meta["equity_source"] == "live"
    assert meta["cap_abs"] is None
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
    assert diag["cap_meta"]["reason"] == "cap_lt_one_lot"


def test_lots_from_premium_cap_fallbacks_to_default_equity(monkeypatch):
    """When live equity is unavailable, the configured default should be used."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.50, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    monkeypatch.setattr(cfg.risk, "default_equity", 40_000.0, raising=False)
    monkeypatch.setattr(cfg.risk, "min_equity_floor", 20_000.0, raising=False)
    lots, meta = lots_from_premium_cap(
        premium=100.0,
        lot_size=50,
        settings_obj=cfg,
        live_equity=None,
    )
    assert meta["equity"] == 40_000.0
    assert meta["cap"] == 20_000.0
    assert lots == 4
    assert meta["source"] == "equity"
    assert meta["equity_source"] == "default"
    assert meta["cap_abs"] is None


def test_lots_from_premium_cap_respects_absolute_cap(monkeypatch):
    """Absolute caps should clamp exposure when configured via ENV source."""
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "env", raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_ABS", 5_000.0, raising=False)
    lots, meta = lots_from_premium_cap(
        premium=100.0,
        lot_size=50,
        settings_obj=cfg,
        live_equity=100_000.0,
    )
    assert meta["cap"] == 5_000.0
    assert meta["unit_notional"] == 5_000.0
    assert meta["source"] == "env"
    assert lots == 1
    assert meta["cap_abs"] == 5_000.0


def test_lots_from_premium_cap_uses_floor_equity(monkeypatch):
    """When defaults are zero, the floor should provide sizing."""

    monkeypatch.setattr(cfg, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.50, raising=False)
    monkeypatch.setattr(cfg, "EXPOSURE_CAP_SOURCE", "equity", raising=False)
    monkeypatch.setattr(cfg.risk, "default_equity", 0.0, raising=False)
    monkeypatch.setattr(cfg.risk, "min_equity_floor", 25_000.0, raising=False)

    lots, meta = lots_from_premium_cap(
        premium=100.0,
        lot_size=25,
        settings_obj=cfg,
        live_equity=None,
    )

    assert meta["equity"] == 25_000.0
    assert meta["cap"] == 12_500.0
    assert lots == 5
    assert meta.get("reason") is None
    assert meta["equity_source"] == "floor"
    assert meta["cap_abs"] is None

    sizer = PositionSizer()
    _, _, diag = sizer.size_from_signal(
        entry_price=100.0,
        stop_loss=90.0,
        lot_size=25,
        equity=25_000.0,
        spot_sl_points=10.0,
        delta=0.5,
    )
    assert diag["cap_meta"]["cap"] == 12_500.0

