"""Research replay must not confuse book observations with fills or profit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.reporting.rejected_setup_replay import analyze

IST = timezone(timedelta(hours=5, minutes=30))
SETUP = "smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-27T05:59:00+00:00"
SYMBOL = "NFO:NIFTY26OCT22750CE"


def logged(second: int, message: str) -> str:
    stamp = (
        datetime(2026, 10, 27, 11, 34, 0, tzinfo=IST) + timedelta(seconds=second)
    ).strftime("%Y-%m-%d %H:%M:%S")
    return f"[{stamp} IST] {message}"


def setup(second: int = 5, setup_id: str = SETUP) -> str:
    return logged(
        second,
        "SETUP_LIFECYCLE stage=TRIGGER_QUALIFIED "
        f"strategy=SMC symbol={SYMBOL} side=CE setup_id={setup_id} reason=None",
    )


def book(
    second: int,
    version: int,
    *,
    reason: str = "None",
    aligned: bool = True,
    conflict: bool = False,
) -> str:
    source = "temporal_ofi" if aligned else "None"
    return logged(
        second,
        f"ORDERFLOW_CONTEXT_EVIDENCE symbol={SYMBOL} side=CE eligible=True "
        f"aligned={aligned} conflict={conflict} source={source} ofi_ready=True "
        f"ofi_sustained={aligned} ofi_reversal_holdoff=False "
        f"ofi_updates_3s=4 ofi_3s_span=2.5 age_s=0.01 "
        f"quote_version={version} block_reason={reason}",
    )


def vote(second: int, version: int, setup_id: str = SETUP) -> str:
    return logged(
        second,
        f"STRATEGY_TRIGGER_VOTE strategy=SMC symbol={SYMBOL} "
        f"side=CE setup_id={setup_id} quote_update_version={version}",
    )


def test_deduplicates_triggers_and_qualified_logged_pair() -> None:
    lines = [setup(), setup(5), vote(5, 10), book(5, 10), vote(11, 20), book(11, 20)]
    result = analyze("\n".join(lines))
    assert result["setup_count"] == 1
    assert result["execution_or_pnl_replay_valid"] is False
    sample = result["setups"][0]
    assert sample["strict_logged_pair"] is True
    assert sample["neutral_tolerant_logged_pair"] is True
    assert sample["trigger_vote_lines"] == 2


def test_neutral_research_variant_is_not_a_live_admission() -> None:
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 11),
        book(8, 11, reason="no_independent_flow_confirmation", aligned=False),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["strict_logged_pair"] is False
    assert sample["neutral_tolerant_logged_pair"] is True
    assert sample["neutral_observations"] == 1


def test_adverse_flow_resets_both_arms() -> None:
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 11),
        book(8, 11, reason="adverse_microstructure", aligned=False, conflict=True),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["strict_logged_pair"] is False
    assert sample["neutral_tolerant_logged_pair"] is False
    assert sample["hard_reset_observations"] == 1


def test_three_second_ofi_misalignment_resets_both_arms() -> None:
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 11),
        book(8, 11, reason="ofi_3s_not_aligned", aligned=False),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["strict_logged_pair"] is False
    assert sample["neutral_tolerant_logged_pair"] is False
    assert sample["hard_reset_observations"] == 1


def test_missing_fields_with_none_reason_do_not_make_neutral_proof() -> None:
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 11),
        book(8, 11, reason="None", aligned=False),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["neutral_tolerant_logged_pair"] is False
    assert sample["hard_reset_observations"] == 1


def test_unmatched_throttled_vote_produces_unknown_not_negative() -> None:
    sample = analyze("\n".join([setup(), vote(5, 10), book(5, 10), vote(11, 20)]))[
        "setups"
    ][0]
    assert sample["strict_logged_pair"] is None
    assert sample["neutral_tolerant_logged_pair"] is None
    assert sample["unmatched_votes"] == 1


def test_wrong_option_symbol_does_not_make_evidence() -> None:
    lines = [setup(), vote(5, 10), book(5, 10).replace(SYMBOL, "NFO:OTHERPE")]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["matched_book_observations"] == 0
    assert sample["strict_logged_pair"] is None


def test_two_versions_but_less_than_four_seconds_never_confirm() -> None:
    lines = [setup(), vote(5, 10), book(5, 10), vote(8, 20), book(8, 20)]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["strict_logged_pair"] is False


def test_repeated_same_quote_vote_does_not_create_a_false_missing_row() -> None:
    lines = [
        setup(),
        vote(5, 10),
        vote(5, 10),
        book(5, 10),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["trigger_vote_lines"] == 3
    assert sample["distinct_vote_snapshots"] == 2
    assert sample["unmatched_votes"] == 0
    assert sample["strict_logged_pair"] is True


def test_mismatched_quote_version_is_indeterminate() -> None:
    sample = analyze("\n".join([setup(), vote(5, 10), book(5, 11)]))["setups"][0]
    assert sample["unmatched_votes"] == 1
    assert sample["strict_logged_pair"] is None


def test_stale_or_ineligible_book_resets_both_arms() -> None:
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 11),
        book(8, 11, reason="no_independent_flow_confirmation", aligned=False).replace(
            "eligible=True", "eligible=False"
        ),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["neutral_tolerant_logged_pair"] is False
    assert sample["hard_reset_observations"] == 1


def test_invalid_identity_is_not_promoted() -> None:
    lines = [
        setup(),
        setup(6).replace("side=CE", "side=PE"),
        vote(5, 10),
        book(5, 10),
        vote(11, 20),
        book(11, 20),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["identity_conflict"] is True
    assert sample["strict_logged_pair"] is None

def test_vote_only_smc_setup_is_counted_but_never_assumed_confirmed() -> None:
    fifth = "smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-27T08:43:00+00:00"
    lines = [
        vote(5, 10, setup_id=fifth),
        book(5, 10),
        vote(11, 20, setup_id=fifth),
        book(11, 20),
    ]
    result = analyze("\n".join(lines))
    assert result["setup_count"] == 1
    assert result["vote_only_setup_count"] == 1
    assert result["lifecycle_stage_setup_count"] == 0
    sample = result["setups"][0]
    assert sample["setup_id"] == fifth
    assert sample["lifecycle_stage_observed"] is False
    assert sample["matched_book_observations"] == 2
    assert sample["strict_logged_pair"] is None
    assert sample["neutral_tolerant_logged_pair"] is None


def test_stage_logged_after_first_vote_preserves_first_seen_identity() -> None:
    lines = [vote(5, 10), book(5, 10), setup(8), vote(11, 20), book(11, 20)]
    result = analyze("\n".join(lines))
    assert result["setup_count"] == 1
    assert result["vote_only_setup_count"] == 0
    assert result["lifecycle_stage_setup_count"] == 1
    sample = result["setups"][0]
    assert sample["first_trigger_at_ist"].endswith("11:34:05+05:30")
    assert sample["lifecycle_stage_observed"] is True
    assert sample["strict_logged_pair"] is True


def test_distinct_vote_only_setup_is_not_merged_into_a_logged_setup() -> None:
    fifth = "smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-27T08:43:00+00:00"
    lines = [
        setup(),
        vote(5, 10),
        book(5, 10),
        vote(8, 20, setup_id=fifth),
        book(8, 20),
    ]
    result = analyze("\n".join(lines))
    assert result["setup_count"] == 2
    assert result["lifecycle_stage_setup_count"] == 1
    assert result["vote_only_setup_count"] == 1
    assert len({row["setup_id"] for row in result["setups"]}) == 2


def test_mismatched_vote_only_symbol_is_not_validated_as_same_setup() -> None:
    lines = [
        vote(5, 10),
        vote(8, 11).replace(SYMBOL, "NFO:NIFTY26OCT22700CE"),
    ]
    sample = analyze("\n".join(lines))["setups"][0]
    assert sample["identity_conflict"] is True
    assert sample["strict_logged_pair"] is None

