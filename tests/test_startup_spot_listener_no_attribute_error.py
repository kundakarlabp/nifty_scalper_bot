from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Mapping


def test_startup_spot_listener_sets_flag_without_attribute_error() -> None:
    calls: list[str] = []
    ctx = SimpleNamespace(startup_spot_refresh_done=False)
    loop = asyncio.new_event_loop()

    async def _refresh_readiness_after_first_tick(_: Any, *, reason: str) -> None:
        calls.append(reason)

    policy = SimpleNamespace(nifty_internal_symbol='NSE:NIFTY')

    def _startup_spot_tick_listener(tick: Mapping[str, Any]) -> None:
        try:
            symbol = str(tick.get('symbol') or '').upper()
            if symbol != policy.nifty_internal_symbol:
                return
            if ctx.startup_spot_refresh_done:
                return

            ctx.startup_spot_refresh_done = True

            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    _refresh_readiness_after_first_tick(
                        ctx,
                        reason='first_spot_tick_listener',
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - defensive parity
            raise AssertionError(f'listener raised unexpectedly: {exc}') from exc

    try:
        _startup_spot_tick_listener({'symbol': 'NSE:NIFTY'})
    finally:
        loop.close()

    assert ctx.startup_spot_refresh_done is True
