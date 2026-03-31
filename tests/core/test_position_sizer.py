from __future__ import annotations

from nifty_scalper_bot.core.position_sizer import compute_kelly_fraction, size_quantity


def test_kelly_fraction_basic() -> None:
    kelly = compute_kelly_fraction(0.6, 2.0, 1.0, max_kelly_fraction=0.25)
    assert 0.0 <= kelly <= 0.25


def test_kelly_fraction_halved_below_winrate_floor() -> None:
    high = compute_kelly_fraction(0.6, 2.0, 1.0, max_kelly_fraction=1.0)
    low = compute_kelly_fraction(0.39, 2.0, 1.0, max_kelly_fraction=1.0)
    assert low < high


def test_size_quantity_min_lot_and_cap() -> None:
    qty = size_quantity(10, 0.1, 0.1, 0.1, 0.1, lot_size=1, max_capital_exposure_qty=2)
    assert qty == 1
