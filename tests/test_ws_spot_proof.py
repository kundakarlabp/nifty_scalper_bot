from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nifty_scalper_bot.core.app import _wait_for_ws_spot_proof


class _StubMDM:
    async def wait_for_fresh_spot_tick(self, *_args, **_kwargs):
        return None


def test_ws_spot_proof_does_not_return_synthetic() -> None:
    ctx = SimpleNamespace(market_data_manager=_StubMDM())
    value = asyncio.run(_wait_for_ws_spot_proof(ctx, timeout=0.05))
    assert value is None
