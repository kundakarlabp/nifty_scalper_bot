"""Tests for VWAP Pro expiry-day detection."""

from __future__ import annotations

import time as time_module

import pytest

from nifty_scalper_bot.strategies.elite_strategies import vwap_pro as vwap_pro_module
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy
from nifty_scalper_bot.utils.smart_symbol import WEEKLY_EXPIRY_WEEKDAY


class _DummyIndicatorEngine:
    """Minimal indicator engine stub for VWAP Pro tests."""

    def get_indicators(self, symbol: str, names: object) -> dict[str, object]:
        """Args: symbol, names. Returns: dict[str, object]. Raises: None."""
        return {name: None for name in names}


def _make_struct_time(weekday: int) -> time_module.struct_time:
    """Args: weekday. Returns: time.struct_time. Raises: None."""
    return time_module.struct_time((2025, 1, 7, 10, 0, 0, weekday, 7, -1))


def test_vwap_pro_expiry_day_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    monkeypatch.delenv("NIFTY_EXPIRY_WEEKDAY", raising=False)
    monkeypatch.setattr(
        vwap_pro_module.time_module,
        "localtime",
        lambda: _make_struct_time(WEEKLY_EXPIRY_WEEKDAY),
    )
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyIndicatorEngine())

    assert strategy._is_expiry_day() is True


def test_vwap_pro_expiry_day_out_of_range_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    monkeypatch.setenv("NIFTY_EXPIRY_WEEKDAY", "9")
    monkeypatch.setattr(
        vwap_pro_module.time_module,
        "localtime",
        lambda: _make_struct_time(WEEKLY_EXPIRY_WEEKDAY),
    )
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyIndicatorEngine())

    assert strategy._is_expiry_day() is True
