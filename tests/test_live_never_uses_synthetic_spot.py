from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.app import _wait_for_live_spot_or_raise


class _StubMDM:
    async def wait_for_fresh_spot_tick(self, *_args, **_kwargs):
        return None

    def get_cached_ltp(self, *_args, **_kwargs):
        return 0.0


def test_live_never_uses_synthetic_spot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.delenv('ALLOW_SYNTHETIC_MARKET_DATA', raising=False)
    with pytest.raises(RuntimeError, match='refusing synthetic spot in LIVE mode'):
        asyncio.run(
            _wait_for_live_spot_or_raise(
                SimpleNamespace(market_data_manager=_StubMDM()),
                timeout=0.05,
                configured_mode='LIVE',
            )
        )
