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
    mdm._last_history_import_result_by_symbol = {}
    mdm._last_hydration_result_by_symbol = {}
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


def test_same_minute_rest_overlap_reconciles_instead_of_failing() -> None:
    """Contract corrected (2026-07-20 hydration-deadlock incident fix):
    ingest_historical_ohlc always declares source="historical" to
    CandleEngine.import_history, so a same-minute overlap against an already
    finalized bar is now a deterministic RECONCILIATION (REST is the
    exchange-finalized aggregate), not a batch-fatal conflict. This test
    previously asserted the old behavior (failure_total=1, conflict_total=1,
    status=finalized_candle_conflict) - that was the incident-causing defect
    (a single REST/WS overlap aborted the whole hydration batch)."""
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    mdm.ingest_historical_ohlc(SYMBOL, [_bar(close=1.5)])

    assert mdm._candle_metrics["history_hydration_failure_total"] == 0
    assert mdm._candle_metrics["history_hydration_conflict_total"] == 0
    assert mdm._last_history_import_result["status"] in {
        "success_new_bars",
        "success_idempotent",
    }
    assert mdm.get_ohlc_bars(SYMBOL)[-1]["close"] == 1.5


def test_idempotent_import_is_distinguishable_from_failure() -> None:
    mdm = _mdm()
    row = _bar()

    assert mdm.ingest_historical_ohlc(SYMBOL, [row]) == 1
    assert mdm.ingest_historical_ohlc(SYMBOL, [row]) == 0

    assert mdm._last_history_import_result["status"] == "success_idempotent"
    assert mdm._candle_metrics["history_hydration_success_total"] == 2
    assert mdm._candle_metrics["history_hydration_failure_total"] == 0


def test_reconciled_history_import_updates_projection_to_rest_value() -> None:
    """Contract corrected alongside the conflict test above: a reconciled
    REST overlap DOES change the projection (to the REST/incoming value) -
    it is no longer treated as a failed import that must leave storage
    untouched."""
    mdm = _mdm()

    assert mdm.ingest_historical_ohlc(SYMBOL, [_bar()]) == 1
    mdm.ingest_historical_ohlc(SYMBOL, [_bar(close=1.5)])

    assert mdm.get_ohlc_bars(SYMBOL)[-1]["close"] == 1.5


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


def test_concurrent_hydration_counters_and_result_maps_are_thread_safe() -> None:
    mdm = _mdm()
    symbols = [f"NSE:NIFTY{i}" for i in range(8)]
    barrier = __import__("threading").Barrier(len(symbols))
    errors: list[BaseException] = []

    def import_symbol(index: int, symbol: str) -> None:
        try:
            row = _bar(close=1.5)
            row["symbol"] = symbol
            row["timestamp"] = datetime(2026, 1, 1, 0, index, tzinfo=timezone.utc)
            barrier.wait(timeout=2)
            mdm.import_historical_ohlc(symbol, [row], reason="live_gap_fill")
        except BaseException as exc:  # noqa: BLE001 - test captures worker errors
            errors.append(exc)

    threads = [
        __import__("threading").Thread(target=import_symbol, args=(idx, symbol))
        for idx, symbol in enumerate(symbols)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert errors == []
    assert mdm._candle_metrics["history_hydration_request_total"] == len(symbols)
    assert mdm._candle_metrics["history_hydration_success_total"] == len(symbols)
    assert mdm._candle_metrics["historical_gap_fill_request_total"] == len(symbols)
    assert mdm._candle_metrics["historical_gap_fill_success_total"] == len(symbols)
    for symbol in symbols:
        result = mdm.get_last_history_import_result(symbol)
        assert result is not None
        assert result.symbol == symbol
        assert result.success
        assert result.reason == "live_gap_fill"


def test_import_result_counts_malformed_without_confusing_later_failure() -> None:
    mdm = _mdm()
    original_get = mdm.get_candle_engine

    def boom(symbol):
        engine = original_get(symbol)

        def fail_import(*_a, **_k):
            raise RuntimeError("broker secret payload should not leak")

        engine.import_history = fail_import
        return engine

    mdm.get_candle_engine = boom
    result = mdm.import_historical_ohlc(SYMBOL, [_bar()], reason="hydration")
    assert result.status == "failed_validation"
    assert result.error == "history_import_failed"
    assert result.validation_rejected_rows == 0
    assert result.conflicting_rows == 0
    assert "secret" not in str(result.error)
    assert result.imported_at.tzinfo is not None


def test_reconciled_rest_overlap_reports_idempotent_success() -> None:
    mdm = _mdm()
    assert mdm.import_historical_ohlc(SYMBOL, [_bar()]).success
    reconciled = mdm.import_historical_ohlc(SYMBOL, [_bar(close=1.25)])
    assert reconciled.status == "success_idempotent"
    assert reconciled.conflicting_rows == 0
    assert reconciled.validation_rejected_rows == 0
    assert mdm.get_ohlc_bars(SYMBOL)[-1]["close"] == 1.25


def test_malformed_rows_are_counted_on_validation_failure() -> None:
    mdm = _mdm()
    result = mdm.import_historical_ohlc(
        SYMBOL, [{"symbol": SYMBOL, "timestamp": "bad"}]
    )
    assert result.status == "failed_validation"
    assert result.validation_rejected_rows == 1
    assert result.conflicting_rows == 0
    assert result.final_cache_rows == 0


def test_history_import_new_bar_counter_idempotent_and_rollover() -> None:
    mdm = _mdm()
    mdm._cache_len = 1
    first = mdm.import_historical_ohlc(SYMBOL, [_bar()])
    same = mdm.import_historical_ohlc(SYMBOL, [_bar()])
    row2 = _bar(close=3.0)
    row2["high"] = 3.0
    row2["timestamp"] = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    second = mdm.import_historical_ohlc(SYMBOL, [row2])
    assert first.accepted_rows == 1
    assert same.accepted_rows == 0 and same.idempotent_rows == 1
    assert second.accepted_rows == 1
    assert second.final_cache_rows == 1
