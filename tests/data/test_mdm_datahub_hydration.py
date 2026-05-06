from __future__ import annotations

import pytest
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.data_hub import DataHub


@pytest.mark.asyncio
async def test_mdm_hydrate_symbol_history_ingests_and_returns_rows() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(info=lambda *a, **k: None)
    mdm._canonical_symbol = lambda s: s
    rows = [{'timestamp': '2026-01-01T09:15:00Z', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 1}]
    async def _fetch(*_a, **_k):
        return rows
    mdm.fetch_history = _fetch
    mdm.ingest_historical_ohlc = lambda *_a, **_k: 1
    mdm.update_hydration_status = lambda *_a, **_k: None
    mdm.get_ohlc_bars = lambda *_a, **_k: rows
    out = await mdm.hydrate_symbol_history('NSE:NIFTY')
    assert out == rows


@pytest.mark.asyncio
async def test_datahub_hydrate_delegates_to_mdm() -> None:
    hub = DataHub.__new__(DataHub)
    async def _mdm_hydrate(*_a, **_k):
        return [{'x': 1}]
    hub._mdm = SimpleNamespace(hydrate_symbol_history=_mdm_hydrate)
    out = await hub.hydrate_symbol_history('NSE:NIFTY')
    assert out == [{'x': 1}]
