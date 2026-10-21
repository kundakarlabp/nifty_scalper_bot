"""Tests for SafeATRProvider fallback behavior."""

from __future__ import annotations

from nifty_scalper_bot.indicators.atr_provider import ATRSnapshot, SafeATRProvider


class DummyEngine:
    """Minimal indicator engine stub for ATR tests."""

    def __init__(self) -> None:
        self._histories: dict[str, object] = {}

    def compute_atr(self, symbol: str) -> float | None:
        if symbol == "NIFTY":
            return 12.5
        return None


def test_get_atr_uses_underlying_fallback() -> None:
    engine = DummyEngine()
    provider = SafeATRProvider(engine, max_cache_age=60.0)

    snapshot = provider.get_atr("NIFTY26FEBFUT")

    assert snapshot is not None
    assert snapshot.value == 12.5
    assert snapshot.source == "underlying"


def test_atr_snapshot_exposes_current_atr_alias() -> None:
    snapshot = ATRSnapshot(value=12.5, timestamp=123.0)
    assert snapshot.current_atr == 12.5
