from __future__ import annotations

import logging

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
