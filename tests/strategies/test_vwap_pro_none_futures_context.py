"""VWAPPro must survive an unavailable futures context.

Production 2026-07-29: `Failure in VWAPProStrategy._evaluate_signal: must be
real number, not NoneType` fired 44 times -- once per evaluation. The strategy
therefore contributed NO votes all session, and because the failure was caught
by a broad handler it surfaced only as a silent "no vote".

Root cause: the futures-context correction made futures_vwap_slope and
futures_volume_ratio legitimately None when unavailable (previously conflated
with 0.0). The STRATEGY_NO_VOTE log line still formatted them with %.4f/%.2f,
and the logging call raised TypeError before the strategy could return.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import (
    VWAPProStrategy,
    _fmt_optional,
)


def _indicators(**over):
    base = {
        "vwap": 48.0, "atr": 2.1, "close": 45.0, "open": 45.0,
        "high": 46.0, "low": 44.0, "volume": 1_300_000.0,
        "avg_volume": 1_700_000.0, "spread_pct": 0.3,
        "futures_vwap_slope": None, "futures_volume_ratio": None,
        "direction": "CE", "underlying_direction": "CE",
        "contract_side": "PE", "context_age_seconds": 0.06,
        "underlying_direction_confidence": 0.70,
    }
    base.update(over)
    return base


def _strategy():
    return VWAPProStrategy(config=MagicMock(), indicator_engine=MagicMock())


def test_evaluate_signal_survives_none_futures_context() -> None:
    """THE FIX: a None futures context must not abort the evaluation."""
    _strategy()._evaluate_signal("NFO:NIFTY2680424200PE", _indicators(), 45.0)


def test_evaluate_signal_survives_none_slope_only() -> None:
    _strategy()._evaluate_signal(
        "NFO:NIFTY2680424200PE",
        _indicators(futures_volume_ratio=1.2),
        45.0,
    )


def test_evaluate_signal_survives_none_volume_ratio_only() -> None:
    _strategy()._evaluate_signal(
        "NFO:NIFTY2680424200PE",
        _indicators(futures_vwap_slope=0.0004),
        45.0,
    )


def test_fmt_optional_marks_unavailable_rather_than_faking_zero() -> None:
    """None must be visibly 'unavailable', not silently rendered as 0.0000."""
    assert _fmt_optional(None, 4) == "unavailable"
    assert _fmt_optional(None, 2) == "unavailable"
    # A real zero is still a real value and must be shown as such.
    assert _fmt_optional(0.0, 4) == "0.0000"
    assert _fmt_optional(0.0, 2) == "0.00"


def test_fmt_optional_formats_normal_values() -> None:
    assert _fmt_optional(0.00042, 4) == "0.0004"
    assert _fmt_optional(4.111, 2) == "4.11"


def test_fmt_optional_does_not_raise_on_garbage() -> None:
    assert _fmt_optional("abc", 2) == "unavailable"  # type: ignore[arg-type]
