"""Regression test for selected-option runner-delivery blocker normalization."""

from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def test_side_specific_delivery_blockers_normalize_to_one_primary_blocker() -> None:
    decision = normalize_readiness_blockers(
        [
            "selected_ce_runner_delivery_missing",
            "selected_pe_runner_delivery_missing",
            "selected_option_runner_delivery_missing",
        ]
    )

    assert decision.primary_blocker == "selected_option_runner_delivery_missing"
    assert decision.blocker_list == ["selected_option_runner_delivery_missing"]
