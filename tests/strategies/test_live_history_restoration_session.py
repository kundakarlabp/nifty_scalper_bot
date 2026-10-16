"""Regression checks for LIVE historical/cache candle session integrity."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY26O1923150CE"
IN_SESSION = datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc)
AFTER_CLOSE = datetime(2026, 10, 16, 11, 28, tzinfo=timezone.utc)


def _bar(timestamp: datetime) -> dict:
    return {
        "symbol": SYMBOL,
        "timestamp": timestamp,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10,
    }


def _runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._lock = threading.RLock()
    runner._normalize_symbol = lambda sym: str(sym)
    runner._symbol_history = {}
    runner._last_bar_ts = {}
    runner._data_phase = {}
    runner._indicator_engine = IndicatorEngine()
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._seed_pipeline_store = lambda *_a, **_k: None
    runner._seed_candle_engine_from_history = lambda *_a, **_k: None
    runner._maybe_promote_pending_active_basket = lambda **_k: None
    runner._logger = SimpleNamespace(
        info=lambda *_a, **_k: None,
        exception=lambda *_a, **_k: None,
        debug=lambda *_a, **_k: None,
    )
    return runner


def test_live_runner_reseed_does_not_restore_offsession_candles(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    runner = _runner()
    count = runner.reseed_history_from_bars(
        SYMBOL, [_bar(IN_SESSION), _bar(AFTER_CLOSE)], min_bars=1
    )
    assert count == 1
    assert [bar.start for bar in runner._symbol_history[SYMBOL]] == [IN_SESSION]
    assert runner._indicator_engine.history_count(SYMBOL) == 1
    assert runner._last_bar_ts[SYMBOL] == IN_SESSION


def test_live_disk_cache_restore_excludes_out_of_session_rows(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    runner = _runner()
    runner._history_cache_dir = tmp_path
    rows = [_bar(IN_SESSION), _bar(AFTER_CLOSE)]
    payload = [{**row, "timestamp": row["timestamp"].isoformat()} for row in rows]
    runner._history_cache_path(SYMBOL).write_text(json.dumps(payload), encoding="utf-8")

    restored = runner._load_history_cache(SYMBOL)
    assert [row["timestamp"] for row in restored] == [IN_SESSION]


def test_shadow_disk_cache_preserves_research_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    runner = _runner()
    runner._history_cache_dir = tmp_path
    rows = [_bar(IN_SESSION), _bar(AFTER_CLOSE)]
    payload = [{**row, "timestamp": row["timestamp"].isoformat()} for row in rows]
    runner._history_cache_path(SYMBOL).write_text(json.dumps(payload), encoding="utf-8")

    restored = runner._load_history_cache(SYMBOL)
    assert len(restored) == 2
