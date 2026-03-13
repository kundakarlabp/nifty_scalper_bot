"""Tests for ATRSnapshot compatibility interface."""

from nifty_scalper_bot.indicators.atr_provider import ATRSnapshot


def test_atr_snapshot_exposes_current_atr_alias() -> None:
    snapshot = ATRSnapshot(value=12.5, timestamp=123.0)

    assert snapshot.current_atr == 12.5
