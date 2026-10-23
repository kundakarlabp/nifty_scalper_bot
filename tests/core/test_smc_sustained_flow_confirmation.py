"""Regression tests for separate, sustained order-book evidence."""

import time
from types import SimpleNamespace

from nifty_scalper_bot.core.smc_flow_confirmation import SustainedSMCFlow


def observe(tracker, version, *, mono=100.0, setup="sweep-A", **flags):
    fields = {
        "context_quality_eligible": True,
        "effective_context_alignment": True,
        "effective_context_conflict": False,
        "context_alignment_source": "temporal_ofi",
        "ofi_ready": True,
        "ofi_sustained_confirmation": True,
        "ofi_3s_complete": True,
        "ofi_reversal_holdoff_active": False,
        "quote_update_version": version,
        "ofi_update_count_3s": 5,
        "ofi_3s_span_seconds": 2.5,
        "vote_timestamp": time.time(),
    }
    fields.update(flags)
    evidence = SimpleNamespace(strategy="OrderFlow", side="CE", metadata=fields)
    return tracker.observe(
        setup_id=setup,
        symbol="NFO:NIFTY26OCT22650CE",
        side="CE",
        votes=[(None, evidence)],
        monotonic_now=mono,
    )


def test_distinct_book_versions_and_time_separation():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert observe(tracker, 20, mono=106.0)


def test_duplicate_version_is_not_independent():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 10, mono=106.0)


def test_short_interval_cannot_satisfy_persistence():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=101.0)


def test_expired_confirmation_gap():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=114.0)


def test_adverse_flow_resets_confirmation():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 15, mono=105.0, effective_context_conflict=True)
    assert not observe(tracker, 20, mono=110.0)
    assert observe(tracker, 30, mono=115.0)


def test_reversal_holdoff_resets_confirmation():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=106.0, ofi_reversal_holdoff_active=True)
    assert not observe(tracker, 30, mono=112.0)


def test_three_second_coverage_mandatory():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=106.0, ofi_3s_complete=False)


def test_stale_book_timestamp_or_missing_quote():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=106.0, vote_timestamp=1.0)


def test_distinct_structural_setup_identity():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10, setup="sweep-A")
    assert not observe(tracker, 20, mono=106.0, setup="sweep-B")


def test_depth_without_temporal_ofi_is_not_sufficient():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, 20, mono=106.0, context_alignment_source="strong_depth")


def test_invalid_quote_version():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert not observe(tracker, -1, mono=106.0)


def test_missing_setup_identity_fails_closed():
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10, setup="")


def test_proven_setup_emits_exactly_one_proof_despite_more_versions() -> None:
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert observe(tracker, 20, mono=106.0)
    assert not observe(tracker, 30, mono=111.0)
    assert not observe(tracker, 40, mono=117.0)
    # Adverse or incomplete evidence cannot erase a consumed setup proof.
    assert not observe(tracker, 41, mono=120.0, effective_context_conflict=True)
    assert not observe(tracker, 50, mono=125.0)
    assert not observe(tracker, 60, mono=131.0)
    # A genuinely new setup has its own independent two-book evidence.
    assert not observe(tracker, 70, mono=135.0, setup="sweep-B")
    assert observe(tracker, 80, mono=141.0, setup="sweep-B")


def test_proved_setup_stays_consumed_after_invalid_book_identity() -> None:
    tracker = SustainedSMCFlow()
    assert not observe(tracker, 10)
    assert observe(tracker, 20, mono=106.0)
    assert not observe(tracker, -1, mono=111.0)
    assert not observe(tracker, 30, mono=116.0)
    assert not observe(tracker, 40, mono=122.0)
