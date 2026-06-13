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
