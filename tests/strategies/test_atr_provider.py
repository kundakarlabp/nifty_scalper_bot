'''Tests for SafeATRProvider fallback behavior.'''

from __future__ import annotations

from nifty_scalper_bot.indicators.atr_provider import SafeATRProvider


class DummyEngine:
    '''Minimal indicator engine stub for ATR tests.'''

    def __init__(self) -> None:
        self._histories: dict[str, object] = {}

    def compute_atr(self, symbol: str) -> float | None:
        if symbol == 'NIFTY':
            return 12.5
        return None


def test_get_atr_uses_underlying_fallback() -> None:
    engine = DummyEngine()
    provider = SafeATRProvider(engine, max_cache_age=60.0)

    snapshot = provider.get_atr('NIFTY26FEBFUT')

    assert snapshot is not None
    assert snapshot.value == 12.5
    assert snapshot.source == 'underlying'
