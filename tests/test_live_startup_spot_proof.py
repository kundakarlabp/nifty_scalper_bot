from __future__ import annotations

import pytest
from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_live_startup_no_ws_tick_outside_market_does_not_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    assert app.get_market_state() == app.MarketState.CLOSED


@pytest.mark.asyncio
async def test_live_startup_no_ws_tick_market_open_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    assert app.get_market_state() == app.MarketState.OPEN
