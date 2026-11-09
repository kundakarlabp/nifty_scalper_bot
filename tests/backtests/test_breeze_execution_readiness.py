"""Aggregate historical audit output never upgrades OHLCV to execution evidence."""

import pytest

from scripts.audit_breeze_execution_readiness import audit_reports


def coverage():
    return {
        "input_archives": 25,
        "total_bars": 1454867,
        "total_index_bars": 184786,
        "total_futures_bars": 536695,
        "total_option_bars": 733386,
        "total_contract_days": 3954,
        "all_failed_contracts": [{} for _ in range(19)],
        "all_empty_contracts": [{} for _ in range(63)],
        "all_missing_minute_contracts": [{} for _ in range(176)],
        "all_source_unavailable": [{} for _ in range(29)],
    }


def validation():
    return {"archives":25,"sha256_mismatches":0,"row_issue_count":0}


def test_valid_archives_still_cannot_certify_execution():
    repair={"original_windows":25,"repair_requested":158,"invalid_candle_hashes":0,
            "hash_states":{"identical":139},"net_bar_delta":0,"retry_failed":19}
    report=audit_reports(coverage(), validation(), repair)
    assert report["observed_bars"] == 1454867
    assert report["observed_selected_option_bars"] == 733386
    assert not report["evidence_gate"]["execution_grade_ready"]
    assert not report["promotion_eligible"]
    assert report["repair"]["net_additional_candles"] == 0
    assert "observed_five_level_quotes" in report["evidence_gate"]["missing_evidence"]


@pytest.mark.parametrize("broken", [
    {"archives":24,"sha256_mismatches":0,"row_issue_count":0},
    {"archives":25,"sha256_mismatches":1,"row_issue_count":0},
    {"archives":25,"sha256_mismatches":0,"row_issue_count":1},
])
def test_integrity_mismatch_blocks_readiness(broken):
    with pytest.raises(ValueError,match="historical_archive_integrity_not_verified"):
        audit_reports(coverage(), broken)


def test_backfill_mismatch_rejected():
    with pytest.raises(ValueError,match="targeted_repair_integrity_unverified"):
        audit_reports(
            coverage(), validation(),
            {"original_windows": 24, "repair_requested": 158,
                                                 "invalid_candle_hashes":0})
