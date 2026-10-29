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


def _record(
    tmp_path,
    *,
    quote=True,
    changed=False,
    rotated_quotes=False,
    rotated_ce_depth=True,
    redundant_snapshot=False,
):
    recorder = ReplayArchive(tmp_path)
    t = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
    recorder.record("snapshot", _snapshot(), t)
    for symbol in (CE, PE, FUT, "NSE:NIFTY 50"):
        tick = _tick(symbol, full=True)
        if not quote:
            tick.pop("depth")
        recorder.record("tick", tick, t + timedelta(seconds=1))
    if redundant_snapshot:
        recorder.record("snapshot", _snapshot(), t + timedelta(seconds=2))
    if changed:
        new_ce = "NFO:NIFTY26OCT23200CE"
        recorder.record("snapshot", _snapshot(new_ce), t + timedelta(minutes=1))
        if rotated_quotes:
            for symbol in (new_ce, PE, FUT, "NSE:NIFTY 50"):
                tick = _tick(symbol, full=True)
                if symbol == new_ce and not rotated_ce_depth:
                    tick.pop("depth")
                recorder.record("tick", tick, t + timedelta(minutes=1, seconds=1))
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


def test_single_level_depth_does_not_certify_five_level_option_quotes(tmp_path):
    recorder = ReplayArchive(tmp_path)
    now = datetime(2026, 10, 21, 10, 0, tzinfo=IST)
    recorder.record("snapshot", _snapshot(), now)
    for symbol in (CE, PE, FUT, "NSE:NIFTY 50"):
        recorder.record(
            "tick",
            _tick(symbol, full=symbol not in {CE, PE}),
            now + timedelta(seconds=1),
        )
    recorder.close()

    report = audit_session(tmp_path / "2026-10-21.jsonl")
    assert not report["observed_selected_context_available"]
    assert report["roles"]["CE"]["uncrossed_two_sided_depth_ticks"] == 1
    assert report["roles"]["CE"]["five_level_depth_ticks"] == 0
    assert report["basket_segments_missing_by_requirement"] == {
        "CE_no_five_level_depth": 1,
        "PE_no_five_level_depth": 1,
    }


def test_dynamic_basket_without_new_quote_context_is_blocked(tmp_path):
    report = audit_session(_record(tmp_path, changed=True))
    assert report["basket_versions"] == 2
    assert not report["observed_selected_context_available"]


def test_complete_rotated_baskets_pass_strict_per_basket_quote_fidelity(tmp_path):
    report = audit_session(_record(tmp_path, changed=True, rotated_quotes=True))

    assert report["basket_versions"] == 2
    assert report["basket_segments"] == 2
    assert report["basket_segments_with_quote_context"] == 2
    assert report["observed_selected_context_available"]
    assert report["basket_segments_missing_quote_context"] == 0
    # A successful tape audit still does not establish executable fill parity.
    assert not report["promotion_eligible"]
    assert not report["live_equivalent"]


def test_repeated_same_basket_snapshot_reuses_verified_quote_segment(tmp_path):
    report = audit_session(_record(tmp_path, redundant_snapshot=True))

    assert report["snapshot_count"] == 2
    assert report["basket_versions"] == 1
    assert report["basket_segments"] == 1
    assert report["basket_segments_with_quote_context"] == 1
    assert report["observed_selected_context_available"]


def test_rotated_basket_without_new_quotes_remains_blocked(tmp_path):
    report = audit_session(_record(tmp_path, changed=True))

    assert report["basket_versions"] == 2
    assert report["basket_segments_missing_quote_context"] == 1
    assert not report["observed_selected_context_available"]


def test_second_basket_missing_depth_is_not_rescued_by_first_basket(tmp_path):
    report = audit_session(
        _record(tmp_path, changed=True, rotated_quotes=True, rotated_ce_depth=False)
    )
    assert report["basket_segments_with_quote_context"] == 1
    assert report["basket_segments_missing_quote_context"] == 1
    assert report["basket_segments_missing_by_requirement"] == {
        "CE_no_two_sided_depth": 1
    }
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
        "tick",
        {"symbol": CE, "ltp": 90, "source": "ws"},
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
    assert CE not in json.dumps(result)
    assert PE not in json.dumps(result)
    assert FUT not in json.dumps(result)


def test_on_demand_runtime_runner_automatically_writes_observed_quality(tmp_path):
    from scripts.run_runtime_replay import write_capture_preflight

    recording = tmp_path / "raw"
    recording.mkdir()
    path = _record(recording)
    report = write_capture_preflight(path, tmp_path / "results")
    saved = json.loads((tmp_path / "results" / "capture-fidelity.json").read_text())
    assert saved["archive_integrity_verified"] is True
    assert saved["observed_selected_context_available"] is True
    assert saved["capture_sha256"] == report["capture_sha256"]
    assert saved["account_pnl"] is None


def test_failed_audit_does_not_generate_fictitious_price_or_execution(tmp_path):
    from scripts.run_runtime_replay import write_capture_preflight

    recording = tmp_path / "raw"
    recording.mkdir()
    path = _record(recording)
    path.write_text(path.read_text().splitlines()[0] + "\n")
    report = write_capture_preflight(path, tmp_path / "results")
    assert report["state"] == "unavailable"
    assert report["error_code"] == "replay_capture_fidelity_preflight_unavailable"
    assert report["archive_integrity_verified"] is False
    assert report["promotion_eligible"] is False
    assert report["account_pnl"] is None
    assert "price" not in json.dumps(report)
