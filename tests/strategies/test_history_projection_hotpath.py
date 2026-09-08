from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY26SEPFUT"


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 8, 3, 45, tzinfo=timezone.utc)
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 24_000.0 + index,
            "high": 24_001.0 + index,
            "low": 23_999.0 + index,
            "close": 24_000.5 + index,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def _runner(
    canonical: list[dict[str, object]], projected: list[dict[str, object]]
) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    runner._normalize_symbol = lambda value: str(value)
    runner._symbol_history = {SYMBOL: list(projected)}
    runner._indicator_engine = IndicatorEngine()
    runner._indicator_engine.replace_history(
        SYMBOL, projected, source="test", min_bars=1
    )
    runner._get_mdm_bars = lambda _symbol, limit: list(canonical)[-limit:]
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._schedule_runtime_history_ensure = lambda *_a, **_k: True
    runner._should_log_throttled = lambda *_a, **_k: False
    runner._market_data = SimpleNamespace(
        history_capacity_for=lambda *_a, **_k: len(canonical)
    )
    runner._data_hub = None
    return runner


def test_wait_until_ready_uses_completed_ohlc_depth_not_raw_tick_count() -> None:
    source = inspect.getsource(MarketDataManager.wait_until_ready)
    assert "get_ohlc_bars" in source
    assert "_raw_tick_history" not in source


def test_capped_runner_same_latest_does_not_reseed(monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_SYMBOL_HISTORY_MAX_BARS", "500")
    canonical = _bars(600)
    projected = canonical[-500:]
    runner = _runner(canonical, projected)
    runner._indicator_engine.replace_history(
        SYMBOL, canonical, source="test", min_bars=1
    )

    def _unexpected_reseed(*_a, **_k):
        raise AssertionError(
            "bounded runner depth with the same latest bar must not reseed"
        )

    runner.reseed_history_from_bars = _unexpected_reseed
    result = runner.sync_history_from_mdm(
        SYMBOL,
        required_bars=30,
        reason="steady_capped_projection",
        role="futures_context",
        request_if_short=False,
    )

    assert result.success is True
    assert result.runner_bars == 500
    assert result.indicator_bars == 600


def test_single_missing_tail_bar_is_backfilled_without_full_reseed(monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_SYMBOL_HISTORY_MAX_BARS", "500")
    canonical = _bars(501)
    projected = canonical[:-1]
    runner = _runner(canonical, projected)
    calls: list[tuple[str, bool, datetime]] = []

    def _unexpected_reseed(*_a, **_k):
        raise AssertionError(
            "one advancing canonical bar must use incremental backfill"
        )

    def _ingest(symbol, bar, is_backfill=False):
        calls.append((symbol, bool(is_backfill), bar.timestamp))
        runner._symbol_history.setdefault(symbol, []).append(bar)
        runner._indicator_engine.ingest_historical_bar(symbol, bar.as_mapping())

    runner.reseed_history_from_bars = _unexpected_reseed
    runner._ingest_bar = _ingest

    result = runner.sync_history_from_mdm(
        SYMBOL,
        required_bars=30,
        reason="tail_advance",
        role="futures_context",
        request_if_short=False,
    )

    assert result.success is True
    assert calls == [(SYMBOL, True, canonical[-1]["timestamp"])]
