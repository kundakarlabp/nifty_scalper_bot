"""Indicator readiness regression checks."""

from __future__ import annotations

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_indicator_ready_status_contains_expected_keys() -> None:
    """Check indicator readiness API shape. Args: none. Returns: none. Raises: none."""
    runner = StrategyRunner([])
    status = runner.get_indicator_ready_status('NFO:NIFTY24APR23900CE')
    expected = {
        'runner_bars',
        'latest_bar_time',
        'vwap_ready',
        'atr_ready',
        'bb_ready',
        'smc_ready',
        'orb_ready',
        'volume_ready',
        'missing',
        'reasons',
    }
    assert expected.issubset(status.keys())
