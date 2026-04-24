"""Tests for VWAP Pro data health safeguards."""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.strategies.elite_strategies import vwap_pro as vwap_pro_module
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyIndicatorEngine:
    """Minimal indicator engine stub for VWAP Pro tests."""

    def get_indicators(self, symbol: str, names: Any) -> dict[str, Any]:
        """Args: symbol, names. Returns: dict[str, Any]. Raises: None."""
        return {name: None for name in names}


def _base_indicators(*, volume: float, avg_volume: float) -> dict[str, Any]:
    """Args: volume, avg_volume. Returns: dict[str, Any]. Raises: None."""
    return {
        'vwap': 90.0,
        'atr': 2.0,
        'entropy_5': 0.5,
        'nifty_fut_ltp': 20000.0,
        'nifty_fut_vwap': 19900.0,
        'vwap_std': 0.0,
        'volume': volume,
        'avg_volume': avg_volume,
    }


def test_vwap_pro_rejects_invalid_volume_data() -> None:
    """Args: None. Returns: None. Raises: AssertionError."""
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyIndicatorEngine())
    signal = strategy.generate_signal(
        symbol='NFO:NIFTY2621025750CE',
        indicators=_base_indicators(volume=0.0, avg_volume=0.0),
        current_price=100.0,
        position=None,
    )

    assert signal is None
    assert strategy._telemetry['skipped_data'] == 1


def test_vwap_pro_uses_volume_grace_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyIndicatorEngine())

    monkeypatch.setattr(vwap_pro_module.time_module, 'time', lambda: 1_000.0)
    strategy.generate_signal(
        symbol='NFO:NIFTY2621025750CE',
        indicators=_base_indicators(volume=50.0, avg_volume=200.0),
        current_price=100.0,
        position=None,
    )

    monkeypatch.setattr(vwap_pro_module.time_module, 'time', lambda: 1_050.0)
    signal = strategy.generate_signal(
        symbol='NFO:NIFTY2621025750CE',
        indicators=_base_indicators(volume=0.0, avg_volume=0.0),
        current_price=100.0,
        position=None,
    )

    assert signal is None
    assert strategy._telemetry['skipped_data'] == 0
    assert strategy._telemetry['skipped_volume'] == 2


def test_vwap_pro_reads_configurable_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VWAP_MAX_OPTION_DISTANCE_PCT', '0.18')
    monkeypatch.setenv('VWAP_SLACK_ATR_MULT', '1.5')
    monkeypatch.setenv('VWAP_ALLOW_PULLBACK_ENTRY', '1')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyIndicatorEngine())
    assert strategy._cfg_max_vwap_distance_pct == pytest.approx(0.18)
    assert strategy._cfg_vwap_slack_atr_mult == pytest.approx(1.5)
    assert strategy._cfg_allow_vwap_pullback is True
