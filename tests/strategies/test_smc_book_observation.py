"""SMC vote book samples are observational and never create broker orders."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.signal_identity import (
    _bounded_smc_book_observation,
    finalize_signal_observability,
)

SYMBOL = "NFO:NIFTY26OCT22700CE"
SETUP = "smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-26T08:33:00+00:00"


@pytest.mark.parametrize(
    ("overrides", "expected_state"),
    [
        ({}, "indicator_book_present"),
        ({"quote_update_version": 9}, "quote_version_unproven"),
        ({"quote_update_version": None}, "quote_version_unproven"),
        ({"bid": None}, "book_missing_or_incoherent"),
        ({"bid": -2}, "book_missing_or_incoherent"),
        ({"bid": float("nan")}, "book_missing_or_incoherent"),
        ({"ask": 99}, "book_missing_or_incoherent"),
        ({"ask": "not-a-price"}, "book_missing_or_incoherent"),
    ],
)
def test_missing_mismatched_or_bad_book_never_claims_valid(
    overrides: dict, expected_state: str
) -> None:
    indicators = {"quote_update_version": 8, "bid": 99.5, "ask": 100.0}
    indicators.update(overrides)
    result = _bounded_smc_book_observation(indicators, {"quote_update_version": 8})
    assert result["book_state"] == expected_state
    if expected_state != "indicator_book_present":
        assert result["book_bid"] is None
        assert result["book_ask"] is None


def test_valid_indicator_book_remains_unverified_observation() -> None:
    result = _bounded_smc_book_observation(
        {
            "quote_update_version": 8,
            "bid": 99.5,
            "ask": 100.0,
            "timestamp_ms": 1793014167000,
            "depth": {"buy": [{"price": 99.5}], "sell": [{"price": 100}]},
        },
        {"quote_update_version": 8},
    )
    assert result["book_state"] == "indicator_book_present"
    assert result["book_bid"] == 99.5
    assert result["book_ask"] == 100.0
    assert result["book_tick_ts_ms"] == 1793014167000.0
    assert result["depth_present"] is True
    assert "executable" not in result


def test_observability_emits_companion_event_without_changing_signal(caplog) -> None:
    signal = SimpleNamespace(
        action="BUY",
        symbol=SYMBOL,
        metadata={
            "strategy": "SMC",
            "strategy_name": "SMC",
            "role": "trigger",
            "contract_side": "CE",
            "setup_id": SETUP,
            "setup_candle_timestamp": "2026-10-26T08:38:00+00:00",
        },
    )
    with caplog.at_level(logging.INFO):
        result = finalize_signal_observability(
            signal,
            {"quote_update_version": 8, "bid": 99.5, "ask": 100.0},
            strategy_name="SMC",
            symbol=SYMBOL,
        )
    assert result is signal
    assert result.metadata["quote_update_version"] == 8
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "SMC_QUOTE_OBSERVATION"
    ]
    assert len(events) == 1
    assert events[0].setup_id == SETUP
    assert events[0].book_state == "indicator_book_present"
    assert events[0].book_bid == 99.5
    assert events[0].quote_update_version == 8
    assert (
        len(
            [
                record
                for record in caplog.records
                if getattr(record, "event", "") == "STRATEGY_TRIGGER_VOTE"
            ]
        )
        == 1
    )


def test_unrelated_strategy_has_no_smc_book_event(caplog) -> None:
    signal = SimpleNamespace(
        action="BUY",
        symbol=SYMBOL,
        metadata={
            "strategy": "VWAPPro",
            "role": "trigger",
            "contract_side": "CE",
            "setup_id": "vwap:CE:2026-10-26",
            "latest_bar_ts": "2026-10-26T08:38:00+00:00",
        },
    )
    with caplog.at_level(logging.INFO):
        finalize_signal_observability(
            signal,
            {"quote_update_version": 8, "bid": 99.5, "ask": 100},
            strategy_name="VWAPPro",
            symbol=SYMBOL,
        )
    assert not any(
        getattr(record, "event", "") == "SMC_QUOTE_OBSERVATION"
        for record in caplog.records
    )
