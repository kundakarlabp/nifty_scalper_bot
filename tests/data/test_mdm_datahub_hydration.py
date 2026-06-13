from __future__ import annotations

import asyncio

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

@pytest.mark.asyncio
async def test_mdm_ensure_history_skips_when_cache_sufficient() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    mdm._canonical_symbol = lambda s: s
    rows = [{'timestamp': '2026-01-01T09:15:00Z', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 1}]
    mdm.get_ohlc_bars = lambda *_a, **_k: rows
    async def _fetch(*_a, **_k):
        raise AssertionError('broker fetch should be skipped')
    mdm.fetch_history = _fetch
    result = await mdm.ensure_history('NSE:NIFTY', required_bars=1, reason='test')
    assert result.success is True
    assert result.fetch_requested is False
    assert result.cached_after == 1


@pytest.mark.asyncio
async def test_mdm_ensure_history_coalesces_identical_requests() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    mdm._canonical_symbol = lambda s: s
    mdm._history_inflight = {}
    mdm._history_inflight_lock = asyncio.Lock()
    stored: list[dict] = []
    calls = 0
    rows = [{'timestamp': '2026-01-01T09:15:00Z', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 1}]
    mdm.get_ohlc_bars = lambda *_a, **_k: list(stored)
    async def _fetch(*_a, **_k):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return rows
    mdm.fetch_history = _fetch
    def _ingest(_symbol, new_rows):
        stored.extend(new_rows)
        return len(new_rows)
    mdm.ingest_historical_ohlc = _ingest
    mdm.update_hydration_status = lambda *_a, **_k: None
    first, second = await asyncio.gather(
        mdm.ensure_history('NSE:NIFTY', required_bars=1, target_bars=1, reason='test'),
        mdm.ensure_history('NSE:NIFTY', required_bars=1, target_bars=1, reason='test'),
    )
    assert calls == 1
    assert first.success is True
    assert second.joined_inflight is True


def test_datahub_history_cache_is_compatibility_only() -> None:
    hub = DataHub.__new__(DataHub)
    mdm_rows = [{'timestamp': '2026-01-01T09:15:00Z', 'open': 9, 'high': 9, 'low': 9, 'close': 9, 'volume': 1}]
    hub._mdm = SimpleNamespace(get_ohlc_bars=lambda *_a, **_k: mdm_rows)
    hub._lock = None
    assert hub.get_ohlc_bars('NSE:NIFTY') == mdm_rows
