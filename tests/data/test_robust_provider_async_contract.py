from __future__ import annotations

import pytest

from nifty_scalper_bot.data.robust_provider import RobustDataProvider


class _Broker:
    def positions(self):
        return {"net": [{"tradingsymbol": "NIFTY", "quantity": 1}]}

    def quote(self, symbols):
        return {"data": {symbol: {"last_price": 1.0} for symbol in symbols}}


@pytest.mark.asyncio
async def test_get_positions_preserves_async_public_contract() -> None:
    provider = RobustDataProvider(_Broker())

    positions = await provider.get_positions()

    assert positions == [{"tradingsymbol": "NIFTY", "quantity": 1}]


@pytest.mark.asyncio
async def test_get_quotes_preserves_async_public_contract() -> None:
    provider = RobustDataProvider(_Broker())

    quotes = await provider.get_quotes(["NFO:TEST"])

    assert quotes == {"NFO:TEST": {"last_price": 1.0}}
