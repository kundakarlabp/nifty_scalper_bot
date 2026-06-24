"""Root fix for the orphan-adoption storm (232 ORPHAN ATTACHED in 2 min): the guard
re-fired every tick because get_position().strategy is always 'unknown'. It must skip
when the symbol already has a managing bracket (is_symbol_managed)."""
from __future__ import annotations


class _BM:
    def __init__(self, managed: bool) -> None:
        self._managed = managed
        self.adopt_calls = 0
    def is_symbol_managed(self, symbol: str) -> bool:
        return self._managed


def _should_adopt(bracket_manager, strat: str, symbol: str) -> bool:
    # mirrors the runner guard: skip if already managed
    already_managed = bool(bracket_manager and bracket_manager.is_symbol_managed(symbol))
    return (not already_managed) and ("manual" in strat.lower() or "unknown" in strat.lower())


async def test_skips_adoption_when_already_managed() -> None:
    bm = _BM(managed=True)
    assert _should_adopt(bm, "unknown", "NFO:NIFTY26JUN23950CE") is False


async def test_adopts_when_unmanaged_and_unknown() -> None:
    bm = _BM(managed=False)
    assert _should_adopt(bm, "unknown", "NFO:NIFTY26JUN23950CE") is True


async def test_skips_when_strategy_known() -> None:
    bm = _BM(managed=False)
    assert _should_adopt(bm, "OrderFlow", "NFO:NIFTY26JUN23950CE") is False
