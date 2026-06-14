from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _bars(count: int) -> list[dict]:
    base = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    return [{"timestamp": base + timedelta(minutes=i), "open": 1+i, "high": 2+i, "low": 1+i, "close": 2+i, "volume": i} for i in range(count)]


def _runner(rows: list[dict] | None = None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    r._normalize_symbol = lambda s: str(s)
    r._symbol_history = {}
    r._indicator_engine = IndicatorEngine()
    r._get_mdm_bars = lambda _s, limit: list(rows or [])[-limit:]
    r._set_symbol_hydration_state = lambda *_a, **_k: None
    r._seed_pipeline_store = lambda *_a, **_k: None
    r._seed_candle_engine_from_history = lambda *_a, **_k: None
    r._active_symbols = set(); r._tracked_symbols = set(); r._data_phase = {}; r._last_bar_ts = {}
    r._runtime_history_ensure_inflight = {}
    r._runtime_history_ensure_roles = {}
    r._runtime_history_ensurer = None
    r._request_mdm_hydration = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no broker fetch from sync"))
    return r


def test_mdm_warm_runner_and_indicator_cold_sync_only() -> None:
    r = _runner(_bars(30))
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert result.success
    assert result.runner_bars >= 30
    assert result.indicator_bars >= 30


def test_mdm_warm_indicator_cold_reseeds_without_fetch() -> None:
    r = _runner(_bars(30))
    r._symbol_history["NSE:NIFTY"] = [object()] * 30
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert result.success
    assert result.failure_reason is None


def test_reseed_failure_returns_failed_result() -> None:
    r = _runner(_bars(30))
    def bad_reseed(*_a, **_k):
        raise ValueError("bad rows")
    r.reseed_history_from_bars = bad_reseed  # type: ignore[method-assign]
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert not result.success
    assert result.failure_reason and result.failure_reason.startswith("runner_reseed_failed")


def test_selected_option_classification_exact_and_unknown_false() -> None:
    r = _runner([])
    r._active_selected_ce = "NFO:NIFTY26JUN23600CE"
    r._active_selected_pe = "NFO:NIFTY26JUN23600PE"
    r._selected_ce_symbol = None; r._selected_pe_symbol = None; r._pending_selected_ce = None; r._pending_selected_pe = None
    r._active_contract_basket = None; r._data_hub = None; r._market_data = None
    assert r._is_selected_option_symbol("NFO:NIFTY26JUN23600CE")
    assert not r._is_selected_option_symbol("NFO:NIFTY26JUN23500CE")
    r._active_selected_ce = None; r._active_selected_pe = None
    assert not r._is_selected_option_symbol("NFO:NIFTY26JUN23600CE")


def test_compat_wrapper_delegates_to_canonical_sync() -> None:
    r = _runner(_bars(5))
    r._required_bars_for_symbol = lambda _s: 5
    r._symbol_role_for_runner = lambda _s: "spot_context"
    assert r._sync_history_from_mdm_cache("NSE:NIFTY", required_bars=5, request_if_short=False) >= 5

import asyncio
import logging
import threading


@pytest.mark.asyncio
async def test_runtime_history_scheduler_suppresses_same_size(caplog) -> None:
    r = _runner([])
    calls: list[dict] = []
    release = asyncio.Event()

    async def ensurer(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        await release.wait()

    r._logger = logging.getLogger("test.runtime_history.same")
    r._runtime_history_ensurer = ensurer
    try:
        with caplog.at_level(logging.INFO):
            assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="first", required_bars=30, target_bars=30)
            await asyncio.sleep(0)
            assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="second", required_bars=30, target_bars=30)
        assert len(calls) == 1
        assert r._runtime_history_ensure_inflight["NFO:CE"] == 30
        rec = next(rec for rec in caplog.records if getattr(rec, "event", "") == "CANONICAL_HISTORY_ENSURE_ALREADY_INFLIGHT")
        assert rec.existing_target == 30 and rec.requested_target == 30
    finally:
        release.set()
        await asyncio.sleep(0)
    assert "NFO:CE" not in r._runtime_history_ensure_inflight


@pytest.mark.asyncio
async def test_runtime_history_scheduler_suppresses_smaller_behind_larger() -> None:
    r = _runner([])
    calls: list[dict] = []
    release = asyncio.Event()

    async def ensurer(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        await release.wait()

    r._runtime_history_ensurer = ensurer
    assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="large", required_bars=75, target_bars=75)
    await asyncio.sleep(0)
    assert r._schedule_runtime_history_ensure("NFO:CE", role="option_context", phase="startup", reason="small", required_bars=30, target_bars=30)
    assert len(calls) == 1
    assert r._runtime_history_ensure_inflight["NFO:CE"] == 75
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_runtime_history_scheduler_allows_larger_upgrade_and_role_metadata(caplog) -> None:
    r = _runner([])
    calls: list[dict] = []
    release = asyncio.Event()

    async def ensurer(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        await release.wait()

    r._logger = logging.getLogger("test.runtime_history.upgrade")
    r._runtime_history_ensurer = ensurer
    with caplog.at_level(logging.INFO):
        assert r._schedule_runtime_history_ensure("NFO:CE", role="option_context", phase="dynamic_update", reason="small", required_bars=30, target_bars=30)
        await asyncio.sleep(0)
        assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="large", required_bars=30, target_bars=75)
        await asyncio.sleep(0)
    assert len(calls) == 2
    assert calls[-1]["role"] == "selected_option"
    assert calls[-1]["target_bars"] == 75
    assert r._runtime_history_ensure_inflight["NFO:CE"] == 75
    assert any(getattr(rec, "event", "") == "CANONICAL_HISTORY_ENSURE_UPGRADED" for rec in caplog.records)
    release.set()
    await asyncio.sleep(0)
    assert "NFO:CE" not in r._runtime_history_ensure_inflight


@pytest.mark.asyncio
async def test_runtime_history_scheduler_callback_exception_cleans_and_allows_retry() -> None:
    r = _runner([])
    calls = 0

    async def failing(symbol, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    r._runtime_history_ensurer = failing
    assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="fail", required_bars=30)
    await asyncio.sleep(0.05)
    assert "NFO:CE" not in r._runtime_history_ensure_inflight

    async def ok(symbol, **kwargs):
        nonlocal calls
        calls += 1

    r._runtime_history_ensurer = ok
    assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="retry", required_bars=30)
    await asyncio.sleep(0.05)
    assert calls == 2
    assert "NFO:CE" not in r._runtime_history_ensure_inflight


@pytest.mark.asyncio
async def test_runtime_history_scheduler_scheduling_failure_restores_inflight(monkeypatch) -> None:
    r = _runner([])
    r._runtime_history_ensurer = lambda *a, **k: None

    class BadLoop:
        def create_task(self, _coro):
            raise RuntimeError("create_task failed")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: BadLoop())
    assert not r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="fail", required_bars=30)
    assert r._runtime_history_ensure_inflight == {}

    r._runtime_history_ensure_inflight["NFO:CE"] = 30
    r._runtime_history_ensure_roles["NFO:CE"] = "option_context"
    assert not r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="upgrade_fail", required_bars=30, target_bars=75)
    assert r._runtime_history_ensure_inflight["NFO:CE"] == 30
    assert r._runtime_history_ensure_roles["NFO:CE"] == "option_context"


@pytest.mark.asyncio
async def test_runtime_history_scheduler_thread_start_failure_cleans(monkeypatch) -> None:
    r = _runner([])
    r._runtime_history_ensurer = lambda *a, **k: None
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))

    class BadThread:
        def __init__(self, *a, **k):
            pass
        def start(self):
            raise RuntimeError("thread failed")

    monkeypatch.setattr(threading, "Thread", BadThread)
    assert not r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="thread_fail", required_bars=30)
    assert r._runtime_history_ensure_inflight == {}


@pytest.mark.asyncio
async def test_runtime_history_scheduler_out_of_order_completion_no_stale_marker() -> None:
    r = _runner([])
    calls: list[int] = []
    releases = {30: asyncio.Event(), 75: asyncio.Event()}

    async def ensurer(symbol, **kwargs):
        target = int(kwargs["target_bars"])
        calls.append(target)
        await releases[target].wait()

    r._runtime_history_ensurer = ensurer
    assert r._schedule_runtime_history_ensure("NFO:CE", role="option_context", phase="dynamic_update", reason="small", required_bars=30, target_bars=30)
    await asyncio.sleep(0)
    assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="large", required_bars=30, target_bars=75)
    await asyncio.sleep(0)
    assert calls == [30, 75]
    assert r._runtime_history_ensure_inflight["NFO:CE"] == 75
    releases[75].set()
    await asyncio.sleep(0.05)
    releases[30].set()
    await asyncio.sleep(0.05)
    assert "NFO:CE" not in r._runtime_history_ensure_inflight


@pytest.mark.asyncio
async def test_runtime_history_scheduler_thread_construction_failure_cleans(monkeypatch) -> None:
    r = _runner([])
    r._runtime_history_ensurer = lambda *a, **k: None
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))

    def bad_thread(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr(threading, "Thread", bad_thread)
    assert not r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="thread_construct_fail", required_bars=30)
    assert r._runtime_history_ensure_inflight == {}
    assert r._runtime_history_ensure_roles == {}


@pytest.mark.asyncio
async def test_runtime_history_scheduler_failure_rollback_does_not_overwrite_newer_upgrade(monkeypatch) -> None:
    r = _runner([])
    r._runtime_history_ensurer = lambda *a, **k: None
    r._runtime_history_ensure_inflight["NFO:CE"] = 30
    r._runtime_history_ensure_roles["NFO:CE"] = "option_context"

    class BadLoop:
        def create_task(self, _coro):
            r._runtime_history_ensure_inflight["NFO:CE"] = 100
            r._runtime_history_ensure_roles["NFO:CE"] = "recovery_or_open_position"
            raise RuntimeError("create_task failed after newer upgrade")

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: BadLoop())
    assert not r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="upgrade_fail", required_bars=30, target_bars=75)
    assert r._runtime_history_ensure_inflight["NFO:CE"] == 100
    assert r._runtime_history_ensure_roles["NFO:CE"] == "recovery_or_open_position"


@pytest.mark.asyncio
async def test_runtime_history_scheduler_setter_injects_callback_and_target() -> None:
    r = _runner([])
    calls: list[dict] = []

    async def ensurer(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})

    r.set_runtime_history_ensurer(ensurer)
    assert r._schedule_runtime_history_ensure("NFO:CE", role="selected_option", phase="startup", reason="target", required_bars=30, target_bars=75)
    await asyncio.sleep(0.05)
    assert calls == [{"symbol": "NFO:CE", "role": "selected_option", "phase": "startup", "reason": "target", "required_bars": 30, "target_bars": 75}]
