"""Long-option SL must never end up at/above entry premium.

Regression: an explicit stop derived from the UNDERLYING (e.g. index ~24000) or an
inverted strategy SL produced SL >= entry premium, which the bracket rejected
(protected_price_invalidates_bracket), killing valid entries.
"""
from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


def _runner() -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None, debug=lambda *a, **k: None)
    return r


def _sig(sl, tp):
    return Signal(action="BUY", symbol="NFO:NIFTY2662324050CE", quantity=65,
                  confidence=0.8, reason="OrderFlow", stop_loss=sl, take_profit=tp)


def _validate(sl, tp, entry=117.25, atr=4.0):
    out = _runner()._validate_long_option_geometry(
        signal=_sig(sl, tp), entry_price=entry, entry_side="BUY", atr=atr
    )
    return out.stop_loss, out.take_profit


async def test_sl_above_entry_is_corrected_below() -> None:
    sl, tp = _validate(140.76, 130.0)  # SL above entry, TP above
    assert 0 < sl < 117.25, f"SL must be a sane premium below entry, got {sl}"
    assert tp > 117.25


async def test_underlying_level_sl_becomes_sane_premium_sl() -> None:
    sl, _ = _validate(24000.0, 0.0)  # underlying-level stop misapplied to premium
    assert 0 < sl < 117.25, f"underlying-derived SL must be clamped to premium range, got {sl}"
    # and within a sane distance (not pinned to 0.05)
    assert (117.25 - sl) <= max(117.25 * 0.6, 4.0 * 2.0) + 0.01


async def test_negative_sl_becomes_sane() -> None:
    sl, _ = _validate(-5.0, 0.0)
    assert 0 < sl < 117.25
