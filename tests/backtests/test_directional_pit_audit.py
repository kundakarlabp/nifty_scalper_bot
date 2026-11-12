"""Fail-closed point-in-time option contract, direction and depth research tests."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest
from scripts import audit_directional_point_in_time as pit

T0 = datetime(2025, 3, 6, 9, 31, tzinfo=timezone(timedelta(hours=5, minutes=30)))
D = T0.date()
SHA = "a" * 64
SYMBOL = "NFO:NIFTY25MAR23000CE"


def event():
    return {
        "event_id": "b" * 64,
        "phase": "validation",
        "event_bar_start": (T0 - timedelta(minutes=1)).isoformat(),
        "earliest_entry": T0.isoformat(),
        "future": "NFO:NIFTY_F1",
        "symbol": SYMBOL,
        "side": "CE",
        "archive_instrument": {
            "expiry": "2025-03-12",
            "strike": 23000,
            "lot_size": 75,
            "token": None,
        },
        "historical_master_verified": False,
        "source_clock_verified": False,
    }


def pilot():
    return {
        "promotion_eligible": False,
        "manifest_sha256": "c" * 64,
        "matched_reversal_events": [event()],
        "session_archive_digests": {"2025-03-06": "d" * 64},
    }


def contract():
    return {
        "source_kind": "NSE_POINT_IN_TIME_INSTRUMENT_MASTER",
        "independent_source_verified": True,
        "source_sha256": SHA,
        "source_observed_at": (T0 - timedelta(days=2)).isoformat(),
        "listed_at": (T0 - timedelta(days=4)).isoformat(),
        "symbol": SYMBOL,
        "right": "CE",
        "expiry": "2025-03-12",
        "strike": 23000,
        "token": "123",
        "lot_size": 75,
    }


def direction(*, futures_bias="CE"):
    return {
        "clock_verified": True,
        "futures_roll_identity_verified": True,
        "source_sha256": SHA,
        "futures_contract": {
            "continuous_alias": "NFO:NIFTY_F1",
            "actual_symbol": "NFO:NIFTY25MARFUT",
            "instrument_token": "987",
        },
        "spot": {
            "source": "NSE_SPOT_NIFTY",
            "source_observed_at": T0.isoformat(),
            "independent_source_verified": True,
            "bias": "CE",
            "confidence": 0.9,
        },
        "futures": {
            "source": "NSE_FUTURES_MAR25",
            "source_observed_at": T0.isoformat(),
            "independent_source_verified": True,
            "bias": futures_bias,
            "confidence": 0.9,
        },
    }


def depth():
    return {
        "source_sha256": SHA,
        "symbol": SYMBOL,
        "token": "123",
        "expiry": "2025-03-12",
        "lot_size": 75,
        "observed_at": (T0 + timedelta(milliseconds=200)).isoformat(),
        "bid": [[100 - i * 0.05, 75] for i in range(5)],
        "ask": [[100.2 + i * 0.05, 75] for i in range(5)],
        "source": "ARCHIVED_FULL",
        "full_depth_captured": True,
        "independent_source_verified": True,
        "clock_verified": True,
    }


def proof():
    return {
        "contract": contract(),
        "direction": direction(),
        "strategy_intent": {
            "verified": True,
            "symbol": SYMBOL,
            "side": "CE",
            "strategy": "FailedBreakoutResearch",
            "source_sha256": SHA,
            "emitted_at": T0.isoformat(),
        },
        "entry_books": [depth()],
        "historical_exchange_rate": {
            "independent_source_verified": True,
            "source_sha256": SHA,
            "rate": 0.0003503,
        },
    }


def audited(evidence=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    return pit.audit(pilot(), evidence or {})


def test_missing_historical_pit_and_quotes_are_unresolved_not_zero_pnl(monkeypatch):
    result = audited(monkeypatch=monkeypatch)
    row = result["results"][0]
    assert row["contract_status"] == "HISTORICAL_EXCHANGE_MASTER_MISSING"
    assert row["underlying_status"] == "UNDERLYING_SOURCE_PROVENANCE_MISSING"
    assert row["entry_status"] == "NOT_ATTEMPTED_MISSING_PROOF"
    assert row["modeled_entry_price"] is None
    assert result["execution_grade_ready"] is False
    assert result["complete_modeled_round_trip_count"] == 0
    assert not row["live_eligible"]


def test_source_master_must_precede_event_and_match_archived_label(monkeypatch):
    evidence = proof()
    evidence["contract"]["source_observed_at"] = (T0 + timedelta(days=1)).isoformat()
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["contract_status"] == (
        "POINT_IN_TIME_CONTRACT_PROOF_INVALID"
    )
    evidence["contract"]["source_observed_at"] = T0.isoformat()
    evidence["contract"]["strike"] = 22950
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["contract_status"] == "ARCHIVED_CONTRACT_LABEL_CONFLICT"


def test_underlying_disagreement_and_unverified_roll_fail_closed(monkeypatch):
    evidence = proof()
    evidence["direction"] = direction(futures_bias="PE")
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["underlying_status"] == (
        "FRESH_SPOT_FUTURES_CONFLICT"
    )
    assert result["results"][0]["modeled_entry_price"] is None
    evidence["direction"] = direction()
    evidence["direction"]["futures_roll_identity_verified"] = False
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["underlying_status"] == (
        "UNDERLYING_SOURCE_PROVENANCE_UNVERIFIED"
    )


def test_invalid_depth_and_missing_fee_never_create_fill(monkeypatch):
    evidence = proof()
    evidence["entry_books"][0]["bid"] = [[99.0, 75]]
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["modeled_entry_price"] is None
    evidence = proof()
    evidence["historical_exchange_rate"]["independent_source_verified"] = False
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["entry_status"] == (
        "HISTORICAL_EXCHANGE_FEE_UNVERIFIED"
    )


def test_quote_depth_only_yields_conditional_entry_not_broker_pnl(monkeypatch):
    evidence = proof()
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    row = result["results"][0]
    assert row["contract_status"] == "POINT_IN_TIME_CONTRACT_ASSERTED"
    assert row["underlying_status"] == "UNDERLYING_DIRECTION_ALIGNED"
    assert row["entry_status"] == "observed_depth_model_only"
    assert row["modeled_entry_price"] == pytest.approx(100.2)
    assert row["modeled_entry_fee"] > 20
    assert not row["broker_fill_verified"]
    assert not row["net_expectancy_verified"]
    assert not result["execution_grade_ready"]


def test_future_dated_signal_cannot_qualify_as_same_event(monkeypatch):
    evidence = proof()
    evidence["strategy_intent"]["emitted_at"] = (
        T0 + timedelta(minutes=1)
    ).isoformat()
    result = audited({"events": {event()["event_id"]: evidence}}, monkeypatch)
    assert result["results"][0]["entry_status"] == (
        "VERIFIED_STRATEGY_INTENT_MISSING"
    )


def test_reject_corrupt_event_ids_and_live_mode(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="requires_shadow"):
        pit.audit(pilot())
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    bad = deepcopy(pilot())
    bad["matched_reversal_events"].append(event())
    with pytest.raises(ValueError, match="identity_invalid"):
        pit.audit(bad)


def test_missing_event_rows_not_confused_with_zero_events(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    bad = deepcopy(pilot())
    bad.pop("matched_reversal_events")
    with pytest.raises(ValueError, match="roster_missing"):
        pit.audit(bad)
