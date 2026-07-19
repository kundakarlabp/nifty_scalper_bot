from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NSE:NIFTY"


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    mdm._cache_len = 100
    mdm._lock = __import__("threading").RLock()
    mdm._ohlc = defaultdict(lambda: deque(maxlen=100))
    mdm._engines = {}
    mdm._candle_metrics = defaultdict(float)
    mdm._candle_queue_watermarks = {}
    mdm._last_history_import_result = None
    mdm._last_historical_ts = {}
    mdm._tick_queue = asyncio.Queue(maxsize=100)
    mdm._bar_symbol_key = lambda s: str(s)
    mdm._canonical_symbol = lambda s: str(s)
    mdm._min_required_bars = 2
    mdm.update_hydration_status = lambda *_a, **_k: None
    return mdm


def _bar(close: float = 2.0, *, open_price: float = 1.0) -> dict:
    return {
        "symbol": SYMBOL,
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "open": open_price,
        "high": 2,
        "low": 1,
        "close": close,
        "volume": 10,
    }


def test_invalid_historical_ohlc_failure_is_not_conflict_or_ready() -> None:
    mdm = _mdm()
    readiness_calls: list[tuple] = []
    mdm.update_hydration_status = lambda *args, **kwargs: readiness_calls.append(
        (args, kwargs)
    )

    accepted = mdm.ingest_historical_ohlc(SYMBOL, [_bar(open_price=-1)])

    assert accepted == 0
    assert readiness_calls == []
    assert mdm._candle_metrics["history_hydration_failure_total"] == 1
    assert mdm._candle_metrics["history_hydration_conflict_total"] == 0
    assert mdm._last_history_import_result["status"] == "failed_validation"


def test_same_minute_finalized_conflict_counts_failure_and_conflict() -> None:
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar(close=1.5)]) == 0

    assert mdm._candle_metrics["history_hydration_failure_total"] == 1
    assert mdm._candle_metrics["history_hydration_conflict_total"] == 1
    assert mdm._last_history_import_result["status"] == "finalized_candle_conflict"


def test_idempotent_import_is_distinguishable_from_failure() -> None:
    mdm = _mdm()
    row = _bar()

    assert mdm.ingest_historical_ohlc(SYMBOL, [row]) == 1
    assert mdm.ingest_historical_ohlc(SYMBOL, [row]) == 0

    assert mdm._last_history_import_result["status"] == "success_idempotent"
    assert mdm._candle_metrics["history_hydration_success_total"] == 2
    assert mdm._candle_metrics["history_hydration_failure_total"] == 0


def test_failed_history_import_leaves_projection_unchanged() -> None:
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    before = mdm.get_ohlc_bars(SYMBOL)
    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar(close=1.5)]) == 0

    assert mdm.get_ohlc_bars(SYMBOL) == before


def test_projection_divergence_detects_equal_length_ohlcv_mismatch() -> None:
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    mdm._ohlc[SYMBOL][0]["close"] = 999
    mdm._refresh_candle_projection(SYMBOL)

    assert mdm._candle_metrics["candle_projection_divergence_total"] == 1


def test_history_results_are_per_symbol() -> None:
    mdm = _mdm()
    other = "NSE:BANKNIFTY"
    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    row = _bar()
    row["symbol"] = other
    assert mdm.ingest_historical_ohlc(other, [row]) == 1

    assert mdm._last_history_import_result_by_symbol[SYMBOL].success
    assert mdm._last_history_import_result_by_symbol[other].success
    assert set(mdm._last_history_import_result_by_symbol) == {SYMBOL, other}


def test_history_result_getter_returns_per_symbol_snapshot() -> None:
    mdm = _mdm()
    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1

    result = mdm.get_last_history_import_result(SYMBOL)

    assert result is not None
    assert result.symbol == SYMBOL
    assert result.success
    assert result.reason == "hydration"


def test_gap_fill_metrics_only_increment_for_live_gap_fill_reason() -> None:
    mdm = _mdm()
    assert mdm.import_historical_ohlc(
        SYMBOL, [_bar()], reason="startup_bootstrap"
    ).success
    assert mdm._candle_metrics["historical_gap_fill_request_total"] == 0
    row = _bar()
    row["timestamp"] = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    assert mdm.import_historical_ohlc(SYMBOL, [row], reason="live_gap_fill").success

    assert mdm._candle_metrics["historical_gap_fill_request_total"] == 1
    assert mdm._candle_metrics["historical_gap_fill_success_total"] == 1


def test_concurrent_two_symbol_history_result_getter_remains_associated() -> None:
    mdm = _mdm()
    other = "NSE:BANKNIFTY"
    barrier = __import__("threading").Barrier(2)

    def import_symbol(symbol: str, close: float) -> None:
        row = _bar(close=close)
        row["symbol"] = symbol
        barrier.wait(timeout=1)
        mdm.import_historical_ohlc(symbol, [row], reason="hydration")

    threads = [
        __import__("threading").Thread(target=import_symbol, args=(SYMBOL, 2.0)),
        __import__("threading").Thread(target=import_symbol, args=(other, 2.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    nifty = mdm.get_last_history_import_result(SYMBOL)
    bank = mdm.get_last_history_import_result(other)
    assert nifty is not None and nifty.symbol == SYMBOL and nifty.success
    assert bank is not None and bank.symbol == other and bank.success
