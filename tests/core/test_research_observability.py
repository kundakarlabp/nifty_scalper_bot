"""Tests for passive selected CE/PE parity and bounded rejection sampling."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.core.research_observability import (
    record_strategy_rejection_sample,
    sample_selected_option_parity,
)

CE = "NFO:NIFTY26N0323700CE"
PE = "NFO:NIFTY26N0323700PE"
STAMP = datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)


def _bar(volume: int, timestamp: datetime = STAMP) -> dict:
    return {
        "timestamp": timestamp,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": volume,
        "is_complete": True,
    }


def _owner(
    indicator: dict[str, list[dict]],
    canonical: dict[str, list[dict]],
):
    hub = SimpleNamespace(
        get_selected_option_symbols=lambda: (CE, PE),
        get_ohlc_bars=lambda symbol, limit: canonical.get(symbol, [])[-limit:],
    )
    engine = SimpleNamespace(
        get_history=lambda symbol, count, field: indicator.get(symbol, [])[-count:]
    )
    return SimpleNamespace(
        _data_hub=hub,
        _indicator_engine=engine,
        _signal_observability_lock=threading.RLock(),
    )


def test_parity_samples_both_options_once_per_minute(caplog) -> None:
    owner = _owner(
        {CE: [_bar(6_989_060)], PE: [_bar(7_854_730)]},
        {CE: [_bar(6_989_060)], PE: [_bar(7_854_730)]},
    )
    at = STAMP + timedelta(minutes=1, seconds=8)
    with caplog.at_level(logging.INFO):
        result = sample_selected_option_parity(owner, at=at)
        assert sample_selected_option_parity(owner, at=at) == []
    assert len(result) == 2
    assert all(item["status"] == "same_minute_volume_match" for item in result)
    assert {item["symbol"] for item in result} == {CE, PE}
    assert caplog.text.count("OPTION_COMPLETED_VOLUME_PARITY") == 2


def test_parity_does_not_call_zero_or_different_minute_verified() -> None:
    owner = _owner(
        {CE: [_bar(0)], PE: [_bar(400)]},
        {
            CE: [_bar(0)],
            PE: [_bar(400, STAMP + timedelta(minutes=1))],
        },
    )
    status = {
        record["symbol"]: record["status"]
        for record in sample_selected_option_parity(
            owner, at=STAMP + timedelta(minutes=1)
        )
    }
    assert status[CE] == "zero_volume_match_unverified"
    assert status[PE] == "different_completed_minutes"


def test_parity_selects_matching_canonical_minute_not_latest_tail() -> None:
    owner = _owner(
        {CE: [_bar(7_854_730)], PE: [_bar(1_234_560)]},
        {
            CE: [_bar(7_854_730), _bar(1, STAMP + timedelta(minutes=1))],
            PE: [_bar(1_234_560)],
        },
    )
    observed = sample_selected_option_parity(owner, at=STAMP + timedelta(minutes=1))
    assert all(row["status"] == "same_minute_volume_match" for row in observed)


def test_rejections_are_exact_contract_minute_samples_not_fake_setups(caplog) -> None:
    owner = _owner({}, {})
    at = STAMP + timedelta(minutes=2, seconds=4)
    with caplog.at_level(logging.INFO):
        a = record_strategy_rejection_sample(
            owner,
            strategy="SMC",
            symbol=CE,
            reason="underlying_no_liquidity_sweep",
            indicators={"latest_bar_ts": STAMP},
            at=at,
        )
        b = record_strategy_rejection_sample(
            owner,
            strategy="SMC",
            symbol=CE,
            reason="smc_underlying_side_mismatch",
            indicators={},
            at=at,
        )
        c = record_strategy_rejection_sample(
            owner,
            strategy="VWAPPro",
            symbol=PE,
            reason="distance_outside_band",
            indicators={},
            at=at,
        )
    assert a is not None and b is None and c is not None
    assert "setup_id" not in a
    assert a["decision_at"] == at.isoformat()
    assert caplog.text.count("RESEARCH_STRATEGY_REJECTION") == 2
    assert (
        record_strategy_rejection_sample(
            owner,
            strategy="SMC",
            symbol=CE,
            reason="underlying_no_liquidity_sweep",
            indicators={},
            at=at + timedelta(minutes=1),
        )
        is not None
    )


def test_rejection_stores_trend_evidence_without_depth_or_secrets(caplog) -> None:
    owner = _owner({}, {})
    with caplog.at_level(logging.INFO):
        row = record_strategy_rejection_sample(
            owner,
            strategy="VWAPPro",
            symbol=CE,
            reason="distance_outside_band",
            indicators={
                "ltp": 28.35,
                "vwap": 20.39,
                "atr": float("nan"),
                "bid": 28.25,
                "ask": 28.35,
                "selected_ce": CE,
                "quote_update_version": 1177,
                "depth": {"full_book": "never_archive_in_journal"},
                "api_secret": "NOT_FOR_RESEARCH",
            },
            at=STAMP,
        )
    assert row is not None
    assert row["context"]["ltp"] == 28.35
    assert row["context"]["vwap"] == 20.39
    assert row["context"]["selected_ce"] == CE
    assert "atr" not in row["context"]
    assert "depth" not in row["context"]
    assert "api_secret" not in row["context"]
    assert "NOT_FOR_RESEARCH" not in caplog.text
    assert "never_archive_in_journal" not in caplog.text
