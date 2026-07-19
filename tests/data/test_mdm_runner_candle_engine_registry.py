from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NSE:NIFTY"


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = _logger()
    mdm._cache_len = 3
    mdm._lock = threading.RLock()
    mdm._ohlc = defaultdict(lambda: deque(maxlen=3))
    mdm._engines = {}
    mdm._candle_metrics = defaultdict(float)
    mdm._candle_queue_watermarks = {}
    mdm._last_history_import_result = None
    mdm._last_historical_ts = {}
    mdm._bar_symbol_key = lambda s: str(s).strip().upper()
    mdm._canonical_symbol = lambda s: (
        SYMBOL
        if str(s).strip().upper() in {"NIFTY", SYMBOL}
        else str(s).strip().upper()
    )
    mdm._min_required_bars = 1
    mdm.update_hydration_status = lambda *_a, **_k: None
    return mdm


def _runner(mdm: MarketDataManager) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = mdm
    runner._candle_engines = {}
    runner._lock = threading.RLock()
    runner._normalize_symbol = lambda s: mdm._canonical_symbol(s)  # type: ignore[method-assign]
    runner._logger = _logger()
    return runner


def _row(minute: int, close: float = 100.0) -> dict[str, Any]:
    ts = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return {
        "symbol": SYMBOL,
        "timestamp": ts,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 10,
    }


def test_mdm_candle_engine_accessor_canonicalizes_to_single_object() -> None:
    mdm = _mdm()

    first = mdm.get_candle_engine("NIFTY")
    second = mdm.get_candle_engine("nse:nifty")

    assert first is second
    assert list(mdm._engines) == [SYMBOL]


def test_runner_resolves_same_authoritative_engine_as_mdm() -> None:
    mdm = _mdm()
    runner = _runner(mdm)

    engine_from_runner = runner._mirror_authoritative_candle_engine("NIFTY")

    assert engine_from_runner is mdm.get_candle_engine(SYMBOL)
    assert runner._candle_engines[SYMBOL] is engine_from_runner


def test_history_imported_through_mdm_is_visible_to_runner_engine() -> None:
    mdm = _mdm()
    runner = _runner(mdm)

    assert mdm.ingest_historical_ohlc("NIFTY", [_row(0)]) == 1
    engine = runner._mirror_authoritative_candle_engine("NIFTY")

    assert engine is mdm.get_candle_engine(SYMBOL)
    assert engine.get_completed_bars()[-1]["close"] == 100.0
    assert (
        engine.latest_finalized_minute()
        == mdm.get_candle_engine(SYMBOL).latest_finalized_minute()
    )


def test_live_finalization_through_mdm_engine_is_visible_to_runner() -> None:
    mdm = _mdm()
    runner = _runner(mdm)
    engine = mdm.get_candle_engine(SYMBOL)
    start = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)

    assert engine.on_tick({"symbol": SYMBOL, "timestamp": start, "ltp": 100.0}) is None
    finalized = engine.on_tick(
        {"symbol": SYMBOL, "timestamp": start + timedelta(minutes=1), "ltp": 101.0}
    )
    mdm._refresh_candle_projection(SYMBOL)

    assert finalized is not None
    assert runner._mirror_authoritative_candle_engine(SYMBOL) is engine
    assert runner._candle_engines[SYMBOL].get_completed_bars()[-1]["close"] == 100.0
    assert mdm.get_latest_closed_bar(SYMBOL) is not None


def test_normal_projection_append_and_rolling_maxlen_do_not_count_divergence() -> None:
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_row(0, 100.0), _row(1, 101.0)]) == 2
    assert mdm._candle_metrics["candle_projection_divergence_total"] == 0
    assert mdm.ingest_historical_ohlc(SYMBOL, [_row(2, 102.0), _row(3, 103.0)]) == 2

    assert len(mdm.get_ohlc_bars(SYMBOL)) == 3
    assert [bar["close"] for bar in mdm.get_ohlc_bars(SYMBOL)] == [101.0, 102.0, 103.0]
    assert mdm._candle_metrics["candle_projection_divergence_total"] == 0


def test_concurrent_first_use_history_and_engine_access_share_identity() -> None:
    mdm = _mdm()
    barrier = threading.Barrier(2)
    results: dict[str, Any] = {}

    def import_history() -> None:
        barrier.wait(timeout=1)
        results["accepted"] = mdm.ingest_historical_ohlc("nifty", [_row(0, 111.0)])
        results["history_engine"] = mdm.get_candle_engine("NSE:NIFTY")

    def live_access() -> None:
        barrier.wait(timeout=1)
        results["live_engine"] = mdm.get_candle_engine("nse:nifty")

    threads = [
        threading.Thread(target=import_history),
        threading.Thread(target=live_access),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert results["accepted"] == 1
    assert results["history_engine"] is results["live_engine"]
    assert list(mdm._engines) == [SYMBOL]
    assert results["live_engine"].get_completed_bars()[-1]["close"] == 111.0


def test_projection_lag_reports_before_and_after_refresh() -> None:
    mdm = _mdm()
    assert mdm.ingest_historical_ohlc(SYMBOL, [_row(0, 100.0)]) == 1
    engine = mdm.get_candle_engine(SYMBOL)
    engine.import_history(
        __import__("pandas").DataFrame([_row(1, 101.0)]),
        mode="incremental",
        source="historical",
    )

    mdm._refresh_candle_projection(SYMBOL)

    diagnostics = mdm._candle_projection_diagnostics[SYMBOL]
    assert diagnostics["lag_before_refresh_seconds"] == 60.0
    assert diagnostics["lag_before_refresh_bars"] == 1.0
    assert diagnostics["lag_after_refresh_seconds"] == 0.0
    assert diagnostics["lag_after_refresh_bars"] == 0.0
    assert mdm._candle_metrics["candle_projection_lag_before_refresh_bars"] == 1.0
    assert mdm._candle_metrics["candle_projection_lag_after_refresh_bars"] == 0.0


def test_projection_diagnostics_are_per_symbol() -> None:
    mdm = _mdm()
    other = "NSE:BANKNIFTY"
    assert mdm.ingest_historical_ohlc(SYMBOL, [_row(0, 100.0)]) == 1
    other_row = _row(0, 200.0)
    other_row["symbol"] = other
    assert mdm.ingest_historical_ohlc(other, [other_row]) == 1

    assert SYMBOL in mdm._candle_projection_diagnostics
    assert other in mdm._candle_projection_diagnostics
    assert (
        mdm._candle_projection_diagnostics[SYMBOL][
            "refreshed_projection_latest_timestamp"
        ]
        == mdm._candle_projection_diagnostics[other][
            "refreshed_projection_latest_timestamp"
        ]
    )
    assert (
        mdm._candle_projection_diagnostics[SYMBOL]
        is not mdm._candle_projection_diagnostics[other]
    )
