from __future__ import annotations

import asyncio

import pytest

from nifty_scalper_bot.core.vwap_history_prewarm import (
    VwapHistoryPrewarmer,
    prioritized_vwap_symbols,
)

CE = "NFO:NIFTY26N1725250CE"
PE = "NFO:NIFTY26N1725250PE"


def test_prewarm_prioritizes_selected_then_same_expiry_adjacent_strikes():
    basket = {
        CE, PE,
        "NFO:NIFTY26N1725200CE", "NFO:NIFTY26N1725200PE",
        "NFO:NIFTY26N1725300CE", "NFO:NIFTY26N1725300PE",
        "NFO:NIFTY26N1825250CE",  # different expiry; must not warm
    }
    ordered = prioritized_vwap_symbols(CE, PE, basket)
    assert ordered[:2] == (CE, PE)
    assert len(ordered) == 6
    assert all("26N17" in symbol for symbol in ordered)


@pytest.mark.asyncio
async def test_prewarm_deduplicates_requests_and_uses_cooldown():
    clock = [100.0]
    prewarmer = VwapHistoryPrewarmer(
        clock=lambda: clock[0], cooldown_seconds=90, max_inflight=2
    )
    bars = {CE: 41, PE: 41}
    seen: list[tuple[str, int]] = []
    pending = asyncio.Event()

    async def hydrate(symbol: str, target: int) -> object:
        seen.append((symbol, target))
        await pending.wait()
        bars[symbol] = target
        return None

    first = prewarmer.schedule(
        CE, PE, [CE, PE], completed_bars=lambda symbol: bars[symbol], hydrate=hydrate
    )
    assert first == (CE, PE)
    assert prewarmer.schedule(
        CE, PE, [CE, PE], completed_bars=lambda symbol: bars[symbol], hydrate=hydrate
    ) == ()
    await asyncio.sleep(0)
    assert seen == [(CE, 50), (PE, 50)]
    pending.set()
    await asyncio.gather(*prewarmer._inflight.values())
    clock[0] += 91
    assert prewarmer.schedule(
        CE, PE, [CE, PE], completed_bars=lambda symbol: bars[symbol], hydrate=hydrate
    ) == ()
    assert seen == [(CE, 50), (PE, 50)]


@pytest.mark.asyncio
async def test_recent_dynamic_hydration_prevents_immediate_repeat_fetch():
    prewarmer = VwapHistoryPrewarmer(clock=lambda: 110.0)
    prewarmer.note_attempt(CE)

    async def hydrate(symbol: str, target: int) -> object:
        raise AssertionError("should not fetch within cooldown")

    assert prewarmer.schedule(
        CE, PE, [CE], completed_bars=lambda _: 41, hydrate=hydrate
    ) == ()
