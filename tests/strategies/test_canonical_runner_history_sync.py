from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _bars(count: int) -> list[dict]:
    base = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    return [{"timestamp": base + timedelta(minutes=i), "open": 1+i, "high": 2+i, "low": 1+i, "close": 2+i, "volume": i} for i in range(count)]


def _runner(rows: list[dict] | None = None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
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
    r._hydration_attempted_symbols = set()
    r._last_hydration_reason_by_symbol = {}
    r._history_role_for_symbol = lambda _s: "spot_context"
    # sync must not start a broker fetch when request_if_short=False
    r._schedule_runtime_history_ensure = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no scheduling from sync when request_if_short=False"))
    def _reseed(symbol, bars, *, source="", min_bars=0):
        rows_list = list(bars or [])
        r._symbol_history[symbol] = list(rows_list)
        for bar in rows_list:
            try:
                r._indicator_engine.ingest_historical_bar(symbol, bar)
            except Exception:
                pass
        return len(rows_list)
    r.reseed_history_from_bars = _reseed
    return r


async def test_mdm_warm_runner_and_indicator_cold_sync_only() -> None:
    r = _runner(_bars(30))
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert result.success
    assert result.runner_bars >= 30
    assert result.indicator_bars >= 30


async def test_mdm_warm_indicator_cold_reseeds_without_fetch() -> None:
    r = _runner(_bars(30))
    r._symbol_history["NSE:NIFTY"] = [object()] * 30
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert result.success
    assert result.failure_reason is None


async def test_reseed_failure_returns_failed_result() -> None:
    r = _runner(_bars(30))
    def bad_reseed(*_a, **_k):
        raise ValueError("bad rows")
    r.reseed_history_from_bars = bad_reseed  # type: ignore[method-assign]
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="test", role="spot_context", request_if_short=False)
    assert not result.success
    assert result.failure_reason and result.failure_reason.startswith("runner_reseed_failed")


async def test_selected_option_classification_exact_and_unknown_false() -> None:
    r = _runner([])
    r._active_selected_ce = "NFO:NIFTY26JUN23600CE"
    r._active_selected_pe = "NFO:NIFTY26JUN23600PE"
    r._selected_ce_symbol = None; r._selected_pe_symbol = None; r._pending_selected_ce = None; r._pending_selected_pe = None
    r._active_contract_basket = None; r._data_hub = None; r._market_data = None
    assert r._is_selected_option_symbol("NFO:NIFTY26JUN23600CE")
    assert not r._is_selected_option_symbol("NFO:NIFTY26JUN23500CE")
    r._active_selected_ce = None; r._active_selected_pe = None
    assert not r._is_selected_option_symbol("NFO:NIFTY26JUN23600CE")


async def test_compat_wrapper_delegates_to_canonical_sync() -> None:
    r = _runner(_bars(5))
    r._required_bars_for_symbol = lambda _s: 5
    r._symbol_role_for_runner = lambda _s: "spot_context"
    assert r._sync_history_from_mdm_cache("NSE:NIFTY", required_bars=5, request_if_short=False) >= 5


# ---- Canonical scheduling on short history (spec §1/§9) ----

def _runner_for_scheduling():
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    r._normalize_symbol = lambda s: str(s)
    r._symbol_history = {}
    r._indicator_engine = IndicatorEngine()
    r._get_mdm_bars = lambda _s, limit: []  # cold -> short -> schedule
    r._set_symbol_hydration_state = lambda *_a, **_k: None
    r._runtime_history_ensure_inflight = {}
    r._runtime_history_ensure_roles = {}
    r._hydration_attempted_symbols = set()
    r._last_hydration_reason_by_symbol = {}
    r._history_role_for_symbol = lambda _s: "selected_option"
    return r


async def test_short_history_schedules_canonical_ensurer_not_request_hydration() -> None:
    r = _runner_for_scheduling()
    calls = []
    async def _ensurer(symbol, *, role, phase, reason, required_bars=None, target_bars=None):
        calls.append((symbol, role, phase, reason, required_bars))
    r.set_runtime_history_ensurer(_ensurer)
    # DataHub/MDM request_hydration must never be touched.
    r._data_hub = SimpleNamespace(request_hydration=lambda *a, **k: (_ for _ in ()).throw(AssertionError("DataHub request_hydration must not be called")))
    r._market_data = SimpleNamespace(request_hydration=lambda *a, **k: (_ for _ in ()).throw(AssertionError("MDM request_hydration must not be called")))
    result = r.sync_history_from_mdm("NFO:NIFTY26JUN24000CE", required_bars=30, reason="t", role="selected_option", request_if_short=True)
    assert result.success is False
    import asyncio as _asyncio
    for _ in range(5):
        await _asyncio.sleep(0)
        if calls:
            break
    assert len(calls) == 1
    symbol, role, phase, reason, required = calls[0]
    assert symbol == "NFO:NIFTY26JUN24000CE"
    assert role == "selected_option"
    assert phase == "runner_sync"
    assert required == 30


async def test_missing_ensurer_fails_safe_no_request_hydration() -> None:
    r = _runner_for_scheduling()
    r._runtime_history_ensurer = None
    r._data_hub = SimpleNamespace(request_hydration=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fall back")))
    r._market_data = SimpleNamespace(request_hydration=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fall back")))
    # must not raise
    result = r.sync_history_from_mdm("NFO:NIFTY26JUN24000CE", required_bars=30, reason="t", role="selected_option", request_if_short=True)
    assert result.success is False


async def test_duplicate_scheduling_suppressed() -> None:
    r = _runner_for_scheduling()
    n = {"count": 0}
    async def _ensurer(symbol, **kw):
        n["count"] += 1
    r.set_runtime_history_ensurer(_ensurer)
    # pre-mark inflight to simulate an active request
    r._runtime_history_ensure_inflight["NFO:NIFTY26JUN24000CE"] = 30
    r._runtime_history_ensure_roles["NFO:NIFTY26JUN24000CE"] = "selected_option"
    scheduled = r._schedule_runtime_history_ensure("NFO:NIFTY26JUN24000CE", role="selected_option", phase="runner_sync", reason="t", required_bars=30)
    assert scheduled is True  # already inflight counts as scheduled
    assert n["count"] == 0  # callback not invoked again


async def test_reseed_runtime_error_returns_structured_failure() -> None:
    r = _runner_for_scheduling()
    r._get_mdm_bars = lambda _s, limit: _bars(30)
    def _boom(*_a, **_k):
        raise RuntimeError("kaboom")
    r.reseed_history_from_bars = _boom
    r._schedule_runtime_history_ensure = lambda *a, **k: True
    result = r.sync_history_from_mdm("NSE:NIFTY", required_bars=30, reason="t", role="spot_context", request_if_short=False)
    assert result.success is False
    assert "runner_reseed_failed:RuntimeError" in (result.failure_reason or "")
