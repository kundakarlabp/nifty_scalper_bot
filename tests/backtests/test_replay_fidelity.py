"""Regression tests for offline raw market quote fidelity preflight."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.backtesting.replay_fidelity import audit_session
from nifty_scalper_bot.storage.replay_archive import ReplayArchive

IST = ZoneInfo("Asia/Kolkata")
CE = "NFO:NIFTY26OCT23100CE"
PE = "NFO:NIFTY26OCT23100PE"
FUT = "NFO:NIFTY26OCTFUT"


def _snapshot(ce=CE):
    return {
        "instruments": [{"instrument_token": x} for x in (101, 102, 103, 256265)],
        "basket": {
            "selected_ce": ce,
            "selected_pe": PE,
            "futures_symbol": FUT,
        },
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"some_value": True},
        "initial_balance": 100000.0,
        "initial_positions": [],
    }


def _tick(symbol, *, full=True, source="ws"):
    n = 5 if full else 1
    return {
        "symbol": symbol,
        "ltp": 101.0,
        "source": source,
        "instrument_token": 102,
        "depth": {
            "buy": [{"price": 100.0 - i, "quantity": 50} for i in range(n)],
            "sell": [{"price": 102.0 + i, "quantity": 50} for i in range(n)],
        },
    }


def _record(tmp_path, *, quote=True, changed=False):
    recorder = ReplayArchive(tmp_path)
    t = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
    recorder.record("snapshot", _snapshot(), t)
    for symbol in (CE, PE, FUT, "NSE:NIFTY 50"):
        tick = _tick(symbol, full=True)
        if not quote:
            tick.pop("depth")
        recorder.record("tick", tick, t + timedelta(seconds=1))
    if changed:
        recorder.record(
            "snapshot", _snapshot("NFO:NIFTY26OCT23200CE"),
            t + timedelta(minutes=1),
        )
    recorder.close()
    return tmp_path / "2026-10-21.jsonl"


def test_verified_capture_has_anonymous_quote_fidelity_statistics(tmp_path):
    report = audit_session(_record(tmp_path))
    assert report["archive_integrity_verified"]
    assert report["event_count"] == 5
    assert report["snapshot_count"] == 1
    assert report["observed_selected_context_available"]
    assert report["roles"]["CE"]["five_level_depth_ticks"] == 1
    assert report["roles"]["PE"]["uncrossed_two_sided_depth_ticks"] == 1
    assert report["roles"]["SPOT"]["observed_market_minutes"] == 1
    assert report["account_pnl"] is None
    assert not report["promotion_eligible"]
    assert not report["actual_exchange_stream_completeness_proven"]


def test_no_depth_is_reported_not_reshaped_into_executable_fill(tmp_path):
    report = audit_session(_record(tmp_path, quote=False))
    assert not report["observed_selected_context_available"]
    assert report["roles"]["CE"]["uncrossed_two_sided_depth_ticks"] == 0
    assert report["roles"]["PE"]["five_level_depth_ticks"] == 0
    assert report["account_pnl"] is None


def test_dynamic_basket_is_recorded_and_rejects_single_basket_parity(tmp_path):
    report = audit_session(_record(tmp_path, changed=True))
    assert report["basket_versions"] == 2
    assert not report["observed_selected_context_available"]


def test_truncated_capture_blocks_analysis(tmp_path):
    path = _record(tmp_path)
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="replay_capture_incomplete"):
        audit_session(path)


def test_invalid_gap_threshold_fails_closed(tmp_path):
    path = _record(tmp_path)
    with pytest.raises(ValueError, match="replay_audit_invalid_gap"):
        audit_session(path, gap_seconds=0)


def test_symbol_or_depth_missing_cannot_confer_replay_fidelity(tmp_path):
    recorder = ReplayArchive(tmp_path)
    now = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
    recorder.record("snapshot", _snapshot(), now)
    recorder.record(
        "tick", {"symbol": CE, "ltp": 90, "source": "ws"},
        now + timedelta(seconds=1),
    )
    recorder.close()
    report = audit_session(tmp_path / "2026-10-21.jsonl")
    assert not report["observed_selected_context_available"]
    assert report["roles"]["PE"]["observed_ticks"] == 0
    assert report["roles"]["CE"]["positive_ltp_ticks"] == 1


def test_replay_checks_digest_without_exposing_tick_contents(tmp_path):
    path = _record(tmp_path)
    result = audit_session(path)
    assert len(result["capture_sha256"]) == 64
    assert "instruments" not in result
    assert "symbol" not in json.dumps(result)
