from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_first_spot_tick_refreshes_readiness() -> None:
    calls: list[str] = []
    ctx = SimpleNamespace(_startup_spot_refresh_done=False)

    async def _refresh() -> None:
        calls.append('called')

    def _listener(tick: dict[str, object]) -> None:
        if tick.get('symbol') == 'NSE:NIFTY' and not ctx._startup_spot_refresh_done:
            ctx._startup_spot_refresh_done = True
            asyncio.run(_refresh())

    _listener({'symbol': 'NSE:NIFTY'})
    _listener({'symbol': 'NSE:NIFTY'})
    assert ctx._startup_spot_refresh_done is True
    assert calls == ['called']
