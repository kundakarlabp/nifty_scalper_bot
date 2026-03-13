"""Tests for ATRSnapshot interface stability."""

from __future__ import annotations

from nifty_scalper_bot.indicators.atr_provider import ATRSnapshot, SafeATRProvider


class _FloatEngine:
    _histories: dict[str, object] = {}

    def compute_atr(self, symbol: str) -> float:
        return 22.5


class _SnapshotEngine:
    _histories: dict[str, object] = {}

    def compute_atr(self, symbol: str) -> ATRSnapshot:
        return ATRSnapshot(value=18.0, timestamp=1.0, source="test")


def test_atr_snapshot_current_atr_alias() -> None:
    snapshot = ATRSnapshot(value=10.0, timestamp=1.0)

    assert snapshot.current_atr == 10.0


def test_provider_wraps_float_inputs_as_snapshot() -> None:
    provider = SafeATRProvider(_FloatEngine(), max_cache_age=60.0)

    snapshot = provider.get_atr("NIFTY")

    assert snapshot is not None
    assert isinstance(snapshot, ATRSnapshot)
    assert snapshot.current_atr == 22.5


def test_provider_accepts_snapshot_inputs() -> None:
    provider = SafeATRProvider(_SnapshotEngine(), max_cache_age=60.0)

    snapshot = provider.get_atr("NIFTY")

    assert snapshot is not None
    assert snapshot.current_atr == 18.0
