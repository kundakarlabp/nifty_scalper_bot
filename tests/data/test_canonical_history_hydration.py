from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.data_hub import DataHub


def _rows(count: int, *, malformed: bool = False) -> list[dict]:
    base = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    rows = []
    for i in range(count):
        if malformed and i % 2:
            rows.append({"bad": i})
        else:
            rows.append({"timestamp": base + timedelta(minutes=i), "open": 1+i, "high": 2+i, "low": 1+i, "close": 2+i, "volume": i})
    return rows


def _mdm(stored: list[dict] | None = None) -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    mdm._canonical_symbol = lambda s: "NSE:NIFTY" if str(s) in {"NIFTY", "NSE:NIFTY"} else str(s)
    mdm._history_inflight = {}
    mdm._history_inflight_lock = asyncio.Lock()
    store = {"rows": list(stored or [])}
    mdm.get_ohlc_bars = lambda *_a, **_k: list(store["rows"])
    def ingest(_symbol, rows):
        existing = {r.get("timestamp") for r in store["rows"] if isinstance(r, dict)}
        accepted = 0
        for row in rows:
            if not isinstance(row, dict) or "timestamp" not in row or not {"open", "high", "low", "close"}.issubset(row):
                continue
            if row["timestamp"] in existing:
                continue
            store["rows"].append(dict(row)); existing.add(row["timestamp"]); accepted += 1
        return accepted
    mdm.ingest_historical_ohlc = ingest
    mdm.update_hydration_status = lambda *_a, **_k: None
    mdm._test_store = store
    return mdm


@pytest.mark.asyncio
async def test_cache_sufficient_for_target_skips_broker() -> None:
    mdm = _mdm(_rows(300))
    async def fetch(*_a, **_k):
        raise AssertionError("no broker fetch")
    mdm.fetch_history = fetch
    result = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="t")
    assert result.minimum_ready and result.target_ready
    assert not result.broker_fetch_started


@pytest.mark.asyncio
async def test_minimum_met_but_target_cold_fetches_target() -> None:
    mdm = _mdm(_rows(30))
    calls = 0
    async def fetch(*_a, **_k):
        nonlocal calls
        calls += 1
        return _rows(300)
    mdm.fetch_history = fetch
    result = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="t")
    assert calls == 1
    assert result.minimum_ready and result.target_ready
    assert result.fetched_rows == 300


@pytest.mark.asyncio
async def test_minimum_only_skips_when_minimum_met() -> None:
    mdm = _mdm(_rows(30))
    async def fetch(*_a, **_k):
        raise AssertionError("minimum-only should skip")
    mdm.fetch_history = fetch
    result = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="t", minimum_only=True)
    assert result.minimum_ready and not result.target_ready
    assert not result.broker_fetch_started


@pytest.mark.asyncio
async def test_identical_concurrent_requests_one_broker_call() -> None:
    mdm = _mdm([])
    calls = 0
    async def fetch(*_a, **_k):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return _rows(30)
    mdm.fetch_history = fetch
    a, b = await asyncio.gather(
        mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=30, reason="a"),
        mdm.ensure_history("NIFTY", required_bars=30, target_bars=30, reason="b"),
    )
    assert calls == 1
    assert a.minimum_ready and b.minimum_ready
    assert b.joined_inflight or a.joined_inflight


@pytest.mark.asyncio
async def test_larger_inflight_satisfies_smaller_request() -> None:
    mdm = _mdm([])
    calls = 0
    async def fetch(*_a, **_k):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return _rows(300)
    mdm.fetch_history = fetch
    big = asyncio.create_task(mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="big"))
    await asyncio.sleep(0)
    small = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=30, reason="small")
    await big
    assert calls == 1
    assert small.minimum_ready


@pytest.mark.asyncio
async def test_smaller_inflight_then_larger_request_fetches_non_overlapping_second_call() -> None:
    mdm = _mdm([])
    calls: list[int] = []
    async def fetch(*_a, **_k):
        calls.append(1)
        await asyncio.sleep(0.01)
        return _rows(30 if len(calls) == 1 else 300)
    mdm.fetch_history = fetch
    small = asyncio.create_task(mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=30, reason="small"))
    await asyncio.sleep(0)
    large = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="large")
    await small
    assert len(calls) == 2
    assert large.target_ready


@pytest.mark.asyncio
async def test_failed_request_cleans_inflight_and_can_retry() -> None:
    mdm = _mdm([])
    calls = 0
    async def fetch(*_a, **_k):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return _rows(30)
    mdm.fetch_history = fetch
    failed = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=30, reason="fail")
    assert failed.failure_reason
    assert mdm._history_inflight == {}
    retry = await mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=30, reason="retry")
    assert retry.minimum_ready


@pytest.mark.asyncio
async def test_independent_ce_pe_concurrency_and_malformed_duplicate_rows() -> None:
    mdm = _mdm([])
    async def fetch(symbol, *_a, **_k):
        return _rows(35, malformed=str(symbol).endswith("CE")) + _rows(5)
    mdm.fetch_history = fetch
    ce, pe = await asyncio.gather(
        mdm.ensure_history("NFO:CE", required_bars=10, target_bars=30, reason="ce"),
        mdm.ensure_history("NFO:PE", required_bars=10, target_bars=30, reason="pe"),
    )
    assert ce.minimum_ready and pe.minimum_ready
    assert ce.accepted_rows <= ce.fetched_rows


@pytest.mark.asyncio
async def test_datahub_facade_delegates_and_has_no_authoritative_cache() -> None:
    mdm = _mdm(_rows(5))
    async def fetch(*_a, **_k):
        return _rows(30)
    mdm.fetch_history = fetch
    hub = DataHub.__new__(DataHub)
    hub._mdm = mdm
    hub._canonical_quote_symbol = lambda s: str(s)
    hub._touch_warm_symbol_cache = lambda *_a, **_k: None
    assert not hasattr(hub, "_history_cache")
    rows = await hub.hydrate_symbol_history("NSE:NIFTY", max_bars=30)
    assert len(rows) >= 30
    rows2 = await hub.fetch_history("NSE:NIFTY", "minute", target_bars=30)
    assert rows2 == rows
