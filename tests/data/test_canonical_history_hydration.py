from __future__ import annotations

import asyncio
import time
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


# ---- DataHub compatibility-facade semantics (spec §1/§3/§11 DataHub) ----

def _datahub_over(mdm: MarketDataManager) -> DataHub:
    hub = DataHub.__new__(DataHub)
    hub._mdm = mdm
    hub._canonical_quote_symbol = lambda s: str(s)
    hub.get_ohlc_bars = lambda *a, **k: list(mdm._test_store["rows"])
    hub._touch_warm_symbol_cache = lambda *a, **k: None
    return hub


async def test_datahub_fetch_history_days_does_not_become_target(monkeypatch) -> None:
    monkeypatch.delenv("DATAHUB_DEFAULT_HISTORY_TARGET", raising=False)
    mdm = _mdm(_rows(60))
    captured = {}
    async def ensure(symbol, **kw):
        captured.update(kw)
        return SimpleNamespace(symbol=symbol, failure_reason=None)
    mdm.ensure_history = ensure
    hub = _datahub_over(mdm)
    await hub.fetch_history("NSE:NIFTY", "minute", days=5)
    # days=5 must NOT make target 5*375=1875; modest default (60) applies.
    assert captured["target_bars"] <= 100, captured
    assert captured["required_bars"] <= captured["target_bars"]
    assert captured["days"] == 5  # days still forwarded as lookback


async def test_datahub_fetch_history_preserves_explicit_deep_target() -> None:
    mdm = _mdm(_rows(10))
    captured = {}
    async def ensure(symbol, **kw):
        captured.update(kw)
        return SimpleNamespace(symbol=symbol, failure_reason=None)
    mdm.ensure_history = ensure
    hub = _datahub_over(mdm)
    await hub.fetch_history("NSE:NIFTY", "minute", days=2, target_bars=300)
    assert captured["target_bars"] == 300


async def test_datahub_hydrate_max_bars_is_ceiling_not_required() -> None:
    mdm = _mdm(_rows(10))
    captured = {}
    async def ensure(symbol, **kw):
        captured.update(kw)
        return SimpleNamespace(symbol=symbol, failure_reason=None)
    mdm.ensure_history = ensure
    hub = _datahub_over(mdm)
    await hub.hydrate_symbol_history("NFO:NIFTYCE", max_bars=300)
    assert captured["target_bars"] == 300
    assert captured["required_bars"] < 300  # not forced to ceiling


async def test_datahub_missing_canonical_api_returns_empty() -> None:
    mdm = SimpleNamespace()  # no ensure_history
    hub = DataHub.__new__(DataHub)
    hub._mdm = mdm
    hub._canonical_quote_symbol = lambda s: str(s)
    rows = await hub.fetch_history("NSE:NIFTY", "minute", days=2)
    assert rows == []


# ---- Joined-request semantics (spec §7/§11 MDM 8-10) ----

async def test_joined_only_caller_reports_not_started() -> None:
    mdm = _mdm(_rows(0))
    started = asyncio.Event()
    release = asyncio.Event()
    async def slow_fetch(*_a, **_k):
        started.set()
        await release.wait()
        return _rows(300)
    mdm.fetch_history = slow_fetch
    # First caller starts the broker task (target 300).
    first = asyncio.ensure_future(mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="first"))
    await asyncio.wait_for(started.wait(), 1.0)
    # Second caller joins the in-flight sufficient request (also target 300).
    second = asyncio.ensure_future(mdm.ensure_history("NSE:NIFTY", required_bars=30, target_bars=300, reason="joiner"))
    await asyncio.sleep(0.05)
    release.set()
    r1 = await asyncio.wait_for(first, 1.0)
    r2 = await asyncio.wait_for(second, 1.0)
    assert r1.broker_fetch_started is True       # creator started it
    assert r2.broker_fetch_started is False      # joiner did not
    assert r2.broker_fetch_observed is True      # joiner observed it
    assert r2.joined_inflight is True


# ---- MDM wrapper: no implicit 300 default (spec §1/§12) ----

async def test_mdm_hydrate_omitted_target_uses_modest_default(monkeypatch) -> None:
    monkeypatch.delenv("MDM_COMPAT_HISTORY_TARGET", raising=False)
    mdm = _mdm(_rows(0))
    captured = {}
    async def ensure(symbol, **kw):
        captured.update(kw)
        return SimpleNamespace(symbol=symbol, failure_reason=None)
    mdm.ensure_history = ensure
    await mdm.hydrate_symbol_history("NFO:NIFTYCE")  # max_bars omitted
    assert captured["target_bars"] <= 100, captured  # modest default, NOT 300
    assert captured["required_bars"] <= captured["target_bars"]


async def test_mdm_hydrate_explicit_max_bars_maps_to_target(monkeypatch) -> None:
    mdm = _mdm(_rows(0))
    captured = {}
    async def ensure(symbol, **kw):
        captured.update(kw)
        return SimpleNamespace(symbol=symbol, failure_reason=None)
    mdm.ensure_history = ensure
    await mdm.hydrate_symbol_history("NFO:NIFTYCE", max_bars=300)
    assert captured["target_bars"] == 300
    assert captured["required_bars"] < 300  # required stays modest

# ---- Raw tick vs canonical OHLC storage/readiness semantics ----

from collections import defaultdict, deque
import threading


def _storage_mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None, debug=lambda *a, **k: None)
    mdm._lock = threading.RLock()
    mdm._min_required_bars = 2
    mdm._raw_tick_history = defaultdict(lambda: deque(maxlen=100))
    mdm._ohlc = defaultdict(lambda: deque(maxlen=100))
    mdm._last_historical_ts = {}
    mdm._bar_symbol_key = lambda s: str(s)
    mdm._canonical_symbol = lambda s: str(s)
    mdm._active_subscribed_symbols = set()
    mdm._readiness_requirements = {}
    mdm._last_readiness_state = {}
    mdm._spot_ready_logged = False
    mdm.update_hydration_status = lambda *_a, **_k: None
    # tick ingestion support
    mdm._symbol_by_token = {}
    mdm._token_to_symbol = {}
    mdm._symbol_to_token = {}
    mdm._token_by_symbol = {}
    mdm._latest_ticks = {}
    mdm._tick_cache = {}
    mdm._last_tick_time = {}
    mdm._ticks_received_per_symbol = defaultdict(int)
    mdm._symbols_with_tick = set()
    mdm._last_tick_wallclock = {}
    mdm._last_quote_ts_ms = {}
    mdm._tick_wallclock = lambda tick: tick.get("timestamp")
    mdm._now_ms = lambda: 1
    mdm._dedupe_symbol_history = lambda *_a, **_k: None
    mdm._refresh_ohlc_from_tick = lambda *_a, **_k: None
    return mdm


def test_historical_bar_writes_only_canonical_ohlc_not_raw_ticks() -> None:
    mdm = _storage_mdm()
    bar = {"symbol": "NSE:NIFTY", "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc), "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}
    mdm.ingest_historical_bar(bar)
    assert len(mdm._ohlc["NSE:NIFTY"]) == 1
    assert len(mdm._raw_tick_history["NSE:NIFTY"]) == 0
    assert mdm.is_ohlc_ready("NSE:NIFTY", required_bars=1) is True
    assert mdm.is_tick_ready("NSE:NIFTY") is False


def test_feed_health_uses_selected_options_not_context_option_staleness() -> None:
    mdm = _storage_mdm()
    now = time.time()
    selected_ce = "NFO:NIFTY2671423950CE"
    selected_pe = "NFO:NIFTY2671423950PE"
    stale_context = "NFO:NIFTY2671424100PE"
    mdm._tick_stale_threshold_ms = 60_000
    mdm._resolve_symbol_key_safe = lambda symbol: str(symbol)
    mdm._active_subscribed_symbols = {selected_ce, selected_pe, stale_context}
    mdm._last_tick_wallclock = {
        "NSE:NIFTY": now,
        "NFO:NIFTY26JULFUT": now,
        selected_ce: now,
        selected_pe: now,
        stale_context: now - 3600,
    }
    mdm._last_tick_time = dict(mdm._last_tick_wallclock)
    mdm.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="NFO:NIFTY26JULFUT",
        atm_ce_symbol=selected_ce,
        atm_pe_symbol=selected_pe,
        option_symbols=[selected_ce, selected_pe, stale_context],
    )

    health = mdm.trading_feed_health(max_age_ms=60_000)

    assert health["options_fresh"] is True
    assert health["option_symbols"] == [selected_ce, selected_pe]


def test_live_tick_writes_raw_tick_history_without_implying_ohlc_ready() -> None:
    mdm = _storage_mdm()
    tick = {"symbol": "NSE:NIFTY", "ltp": 100.0, "timestamp": 1.0, "source": "ws"}
    mdm._store_tick("NSE:NIFTY", dict(tick))
    mdm._store_tick("NSE:NIFTY", dict(tick, timestamp=2.0, ltp=101.0))
    assert len(mdm._raw_tick_history["NSE:NIFTY"]) == 2
    assert mdm.is_tick_ready("NSE:NIFTY") is True
    assert mdm.is_ohlc_ready("NSE:NIFTY") is False
    assert mdm.is_market_data_ready("NSE:NIFTY") is False
