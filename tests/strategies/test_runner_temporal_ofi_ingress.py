from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.order_flow_evidence import TemporalOfiAccumulator
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY26SEP25000CE"


def _tick(
    *,
    version: int,
    buy: float,
    sell: float,
) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "quote_update_version": version,
        "bid": 100.0,
        "ask": 100.5,
        "depth": {
            "buy": [{"quantity": buy}],
            "sell": [{"quantity": sell}],
        },
    }


def test_runner_tick_ingress_builds_one_second_ofi_before_strategy_eval() -> None:
    runner = object.__new__(StrategyRunner)
    runner._temporal_ofi = TemporalOfiAccumulator()
    runner._logger = logging.getLogger("test_runner_temporal_ofi")

    runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=1, buy=100.0, sell=100.0),
        observed_at=10.0,
    )
    runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=2, buy=140.0, sell=100.0),
        observed_at=10.2,
    )
    snapshot = runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=3, buy=160.0, sell=100.0),
        observed_at=10.4,
    )

    assert snapshot["ofi_ready"] is True
    assert snapshot["ofi_event"] == 20.0
    assert snapshot["ofi_1s"] == 60.0
    assert snapshot["ofi_update_count_1s"] == 2
    assert snapshot["ofi_1s_normalized"] > 0.0
    assert snapshot["ofi_source"] == "runner_datahub_tick_updates"


@pytest.mark.parametrize("source", ["ws", "ws_full"])
def test_runner_uses_same_tick_ws_quote_book_for_temporal_ofi(source: str) -> None:
    """Sparse callbacks may borrow ONLY the matching authoritative FULL book."""
    runner = object.__new__(StrategyRunner)
    runner._temporal_ofi = TemporalOfiAccumulator()
    runner._logger = logging.getLogger("test_runner_temporal_ofi")
    live_quote: dict[str, object] = {}
    runner._data_hub = SimpleNamespace(
        get_quote=lambda symbol, allow_pull=False: (
            dict(live_quote) if symbol == SYMBOL and not allow_pull else None
        )
    )
    snapshot: dict[str, object] = {}
    for version, buy_qty in ((1, 100.0), (2, 140.0), (3, 160.0)):
        ts_ms = 1791500000000 + version * 200
        live_quote.clear()
        live_quote.update(
            {
                **_tick(version=version, buy=buy_qty, sell=100.0),
                "source": source,
                "instrument_token": 123,
                "timestamp_ms": ts_ms,
            }
        )
        snapshot = runner._with_temporal_ofi(
            SYMBOL,
            {
                "symbol": SYMBOL,
                "source": source,
                "instrument_token": 123,
                "quote_update_version": version,
                "timestamp_ms": ts_ms,
                "ltp": 100.0,
            },
            observed_at=10.0 + version * 0.2,
        )

    assert snapshot["ofi_ready"] is True
    assert snapshot["ofi_update_count_1s"] == 2
    assert snapshot["ofi_1s"] == pytest.approx(60.0)
    assert "depth" not in snapshot


@pytest.mark.parametrize("invalid", ["version", "timestamp", "source", "token"])
def test_runner_rejects_mismatched_quote_book_for_temporal_ofi(invalid: str) -> None:
    """A later, stale, polling or foreign-token quote cannot supply live OFI."""
    runner = object.__new__(StrategyRunner)
    runner._temporal_ofi = TemporalOfiAccumulator()
    runner._logger = logging.getLogger("test_runner_temporal_ofi")
    live_quote: dict[str, object] = {}
    runner._data_hub = SimpleNamespace(
        get_quote=lambda symbol, allow_pull=False: dict(live_quote)
    )
    for version, buy_qty in ((1, 100.0), (2, 140.0), (3, 160.0)):
        ts_ms = 1791500000000 + version * 200
        live_quote.clear()
        live_quote.update(
            {
                **_tick(version=version, buy=buy_qty, sell=100.0),
                "source": "rest" if invalid == "source" else "ws",
                "instrument_token": 456 if invalid == "token" else 123,
                "timestamp_ms": ts_ms + (2000 if invalid == "timestamp" else 0),
                "quote_update_version": version + (1 if invalid == "version" else 0),
            }
        )
        snapshot = runner._with_temporal_ofi(
            SYMBOL,
            {
                "symbol": SYMBOL,
                "source": "ws",
                "instrument_token": 123,
                "quote_update_version": version,
                "timestamp_ms": ts_ms,
                "ltp": 100.0,
            },
            observed_at=10.0 + version * 0.2,
        )
        assert snapshot["ofi_ready"] is False
        assert snapshot["ofi_update_count_1s"] == 0


def test_runner_ofi_deduplicates_versions_and_resets_after_gap() -> None:
    runner = object.__new__(StrategyRunner)
    runner._temporal_ofi = TemporalOfiAccumulator()
    runner._logger = logging.getLogger("test_runner_temporal_ofi")

    runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=1, buy=100.0, sell=100.0),
        observed_at=10.0,
    )
    runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=2, buy=140.0, sell=100.0),
        observed_at=10.2,
    )
    ready = runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=3, buy=160.0, sell=100.0),
        observed_at=10.4,
    )
    repeated = runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=3, buy=160.0, sell=100.0),
        observed_at=10.5,
    )
    reset = runner._with_temporal_ofi(
        SYMBOL,
        _tick(version=4, buy=180.0, sell=100.0),
        observed_at=16.0,
    )

    assert repeated["ofi_1s"] == ready["ofi_1s"]
    assert repeated["ofi_update_count_1s"] == 2
    assert reset["ofi_ready"] is False
    assert reset["ofi_update_count_1s"] == 0


def test_repeated_quote_version_expires_ready_ofi_after_gap() -> None:
    accumulator = TemporalOfiAccumulator()
    accumulator.update(
        SYMBOL, _tick(version=1, buy=100, sell=100), update_version=1, observed_at=10.0
    )
    accumulator.update(
        SYMBOL, _tick(version=2, buy=140, sell=100), update_version=2, observed_at=10.2
    )
    ready = accumulator.update(
        SYMBOL, _tick(version=3, buy=160, sell=100), update_version=3, observed_at=10.4
    )
    expired = accumulator.update(
        SYMBOL, _tick(version=3, buy=160, sell=100), update_version=3, observed_at=16.0
    )
    assert ready["ofi_ready"] is True
    assert expired["ofi_ready"] is False
    assert expired["ofi_update_count_1s"] == 0


def test_invalid_best_book_clears_temporal_evidence() -> None:
    accumulator = TemporalOfiAccumulator()
    accumulator.update(
        SYMBOL, _tick(version=1, buy=100, sell=100), update_version=1, observed_at=10.0
    )
    accumulator.update(
        SYMBOL, _tick(version=2, buy=140, sell=100), update_version=2, observed_at=10.2
    )
    ready = accumulator.update(
        SYMBOL, _tick(version=3, buy=160, sell=100), update_version=3, observed_at=10.4
    )
    invalid = _tick(version=4, buy=160, sell=100)
    invalid["depth"] = None
    cleared = accumulator.update(SYMBOL, invalid, update_version=4, observed_at=10.5)
    restored = accumulator.update(
        SYMBOL, _tick(version=5, buy=170, sell=100), update_version=5, observed_at=10.6
    )
    assert ready["ofi_ready"] is True
    assert cleared["ofi_ready"] is False
    assert cleared["ofi_1s"] == 0.0
    assert restored["ofi_update_count_1s"] == 0


def test_duplicate_version_ages_windows_without_creating_flow() -> None:
    accumulator = TemporalOfiAccumulator()
    for version, timestamp, buy in ((1, 10.0, 100), (2, 10.2, 140), (3, 10.4, 160)):
        accumulator.update(
            SYMBOL,
            _tick(version=version, buy=buy, sell=100),
            update_version=version,
            observed_at=timestamp,
        )
    aged = accumulator.update(
        SYMBOL,
        _tick(version=3, buy=160, sell=100),
        update_version=3,
        observed_at=11.5,
    )
    assert aged["ofi_ready"] is False
    assert aged["ofi_event"] == 0.0
    assert aged["ofi_1s"] == 0.0
    assert aged["ofi_update_count_1s"] == 0
    assert aged["ofi_3s"] == 60.0
    expired = accumulator.update(
        SYMBOL,
        _tick(version=3, buy=160, sell=100),
        update_version=3,
        observed_at=13.5,
    )
    assert expired["ofi_3s"] == 0.0


@pytest.mark.parametrize("field", ["bid", "ask", "buy", "sell", "observed_at"])
@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_book_or_clock_cannot_authorize_ofi(field, invalid) -> None:
    accumulator = TemporalOfiAccumulator()
    for version, buy in ((1, 100), (2, 140)):
        accumulator.update(
            SYMBOL,
            _tick(version=version, buy=buy, sell=100),
            update_version=version,
            observed_at=10.0 + version / 10.0,
        )
    tick = _tick(version=3, buy=160, sell=100)
    observed_at = 10.3
    if field in {"bid", "ask"}:
        tick[field] = invalid
    elif field in {"buy", "sell"}:
        tick["depth"][field][0]["quantity"] = invalid
    else:
        observed_at = invalid
    snapshot = accumulator.update(
        SYMBOL,
        tick,
        update_version=3,
        observed_at=observed_at,
    )
    assert snapshot["ofi_ready"] is False
    assert snapshot["ofi_1s_normalized"] == 0.0
    restored = accumulator.update(
        SYMBOL,
        _tick(version=4, buy=170, sell=100),
        update_version=4,
        observed_at=10.4,
    )
    assert restored["ofi_update_count_1s"] == 0


def test_bounded_ofi_never_presents_truncated_window_as_ready() -> None:
    accumulator = TemporalOfiAccumulator(max_events=8)
    for version in range(1, 11):
        snapshot = accumulator.update(
            SYMBOL,
            _tick(version=version, buy=100 + version * 10, sell=100),
            update_version=version,
            observed_at=10.0 + version / 100.0,
        )
    assert snapshot["ofi_ready"] is False
    assert snapshot["ofi_1s_complete"] is False
    assert snapshot["ofi_3s_complete"] is False
    recovered = accumulator.update(
        SYMBOL,
        _tick(version=11, buy=220, sell=100),
        update_version=11,
        observed_at=11.2,
    )
    assert recovered["ofi_update_count_1s"] == 1
    ready = accumulator.update(
        SYMBOL,
        _tick(version=12, buy=230, sell=100),
        update_version=12,
        observed_at=11.3,
    )
    assert ready["ofi_ready"] is True
    assert ready["ofi_1s"] == 30.0
    assert ready["ofi_3s_complete"] is False


def test_late_update_does_not_replace_fresher_ofi_baseline() -> None:
    accumulator = TemporalOfiAccumulator()
    accumulator.update(
        SYMBOL,
        _tick(version=1, buy=100, sell=100),
        update_version=1,
        observed_at=10.0,
    )
    accumulator.update(
        SYMBOL,
        _tick(version=2, buy=140, sell=100),
        update_version=2,
        observed_at=10.2,
    )
    late = accumulator.update(
        SYMBOL,
        _tick(version=1, buy=100, sell=100),
        update_version=1,
        observed_at=10.1,
    )
    current = accumulator.update(
        SYMBOL,
        _tick(version=3, buy=160, sell=100),
        update_version=3,
        observed_at=10.3,
    )
    assert late["ofi_ready"] is False
    assert current["ofi_event"] == 20.0
    assert current["ofi_1s"] == 60.0


def test_sparse_ws_book_does_not_destroy_recent_verified_ofi_history() -> None:
    """Interleaved LTP-only ticks must not erase verified FULL book transitions."""
    accumulator = TemporalOfiAccumulator()
    baseline = _tick(version=1, buy=100, sell=100)
    baseline["source"] = "ws"
    accumulator.update(SYMBOL, baseline, update_version=1, observed_at=10.0)
    sparse = {"source": "ws", "ltp": 100.1}
    unavailable = accumulator.update(SYMBOL, sparse, update_version=2, observed_at=10.1)
    assert unavailable["ofi_ready"] is False
    assert unavailable["ofi_update_count_1s"] == 0
    for version, ts, quantity in ((3, 10.2, 140), (4, 10.4, 160)):
        book = _tick(version=version, buy=quantity, sell=100)
        book["source"] = "ws"
        result = accumulator.update(
            SYMBOL, book, update_version=version, observed_at=ts
        )
    assert result["ofi_ready"] is True
    assert result["ofi_update_count_1s"] == 2
    assert result["ofi_1s"] == pytest.approx(60.0)


def test_poll_quote_never_counts_as_temporal_ws_order_flow() -> None:
    accumulator = TemporalOfiAccumulator()
    first = _tick(version=1, buy=100, sell=100)
    first["source"] = "ws"
    accumulator.update(SYMBOL, first, update_version=1, observed_at=10.0)
    poll = _tick(version=2, buy=500, sell=100)
    poll["source"] = "poll"
    untrusted = accumulator.update(SYMBOL, poll, update_version=2, observed_at=10.1)
    assert untrusted["ofi_ready"] is False
    assert untrusted["ofi_update_count_1s"] == 0
    for version, ts, quantity in ((3, 10.2, 140), (4, 10.4, 160)):
        book = _tick(version=version, buy=quantity, sell=100)
        book["source"] = "ws"
        result = accumulator.update(
            SYMBOL, book, update_version=version, observed_at=ts
        )
    assert result["ofi_ready"] is True
    assert result["ofi_update_count_1s"] == 2
    assert result["ofi_1s"] == pytest.approx(60.0)
