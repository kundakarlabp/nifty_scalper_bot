"""Evaluation and execution admission thresholds have one owner.

The runner pregate and TradeCandidateSelector both read MIN_OPTION_PREMIUM but
defaulted to 20 and 40 respectively, so the intended
evaluation-wider-than-execution split silently collapsed whenever an operator
set the shared variable. Spread was likewise 1.5% / 0.75% / 10% across three
layers. These tests pin the ownership and the two invariants.
"""

from __future__ import annotations

import pytest

from nifty_scalper_bot.config.entry_policy import resolve_entry_policy


@pytest.fixture(autouse=True)
def _clean_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MIN_OPTION_PREMIUM",
        "EVALUATION_MIN_OPTION_PREMIUM",
        "EXECUTION_MIN_OPTION_PREMIUM",
        "MAX_OPTION_PREMIUM",
        "MAX_OPTION_SPREAD_PCT_FOR_EVAL",
        "LIVE_CANDIDATE_MAX_SPREAD_PCT",
        "ORDER_MAX_SPREAD_PCT",
        "SPREAD_MAX_PCT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_reproduce_the_previously_hardcoded_thresholds() -> None:
    policy = resolve_entry_policy()

    assert policy.evaluation_min_premium == 20.0
    assert policy.execution_min_premium == 40.0
    assert policy.evaluation_max_spread_pct == 1.5
    assert policy.execution_max_spread_pct == 0.75
    assert policy.max_premium == 650.0


def test_evaluation_is_never_stricter_than_execution() -> None:
    policy = resolve_entry_policy()

    assert policy.evaluation_min_premium <= policy.execution_min_premium
    assert policy.evaluation_max_spread_pct >= policy.execution_max_spread_pct


def test_shared_legacy_variable_still_configures_both_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing operator configuration must keep its current meaning."""
    monkeypatch.setenv("MIN_OPTION_PREMIUM", "55")
    policy = resolve_entry_policy()

    assert policy.evaluation_min_premium == 55.0
    assert policy.execution_min_premium == 55.0


def test_policies_can_be_configured_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALUATION_MIN_OPTION_PREMIUM", "15")
    monkeypatch.setenv("EXECUTION_MIN_OPTION_PREMIUM", "60")
    policy = resolve_entry_policy()

    assert policy.evaluation_min_premium == 15.0
    assert policy.execution_min_premium == 60.0


def test_inverted_premium_configuration_is_repaired_towards_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution must never admit a premium evaluation refused to look at."""
    monkeypatch.setenv("EVALUATION_MIN_OPTION_PREMIUM", "80")
    monkeypatch.setenv("EXECUTION_MIN_OPTION_PREMIUM", "30")
    policy = resolve_entry_policy()

    assert policy.execution_min_premium == 80.0
    assert policy.evaluation_min_premium <= policy.execution_min_premium


def test_inverted_spread_configuration_tightens_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_OPTION_SPREAD_PCT_FOR_EVAL", "0.4")
    monkeypatch.setenv("LIVE_CANDIDATE_MAX_SPREAD_PCT", "0.9")
    policy = resolve_entry_policy()

    assert policy.execution_max_spread_pct == 0.4
    assert policy.evaluation_max_spread_pct >= policy.execution_max_spread_pct


def test_invalid_values_fall_back_to_defaults_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIN_OPTION_PREMIUM", "not-a-number")
    monkeypatch.setenv("MAX_OPTION_SPREAD_PCT_FOR_EVAL", "")
    policy = resolve_entry_policy()

    assert policy.evaluation_min_premium == 20.0
    assert policy.execution_min_premium == 40.0
    assert policy.evaluation_max_spread_pct == 1.5


def test_quality_spread_limit_follows_execution_policy_not_a_ten_percent_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nifty_scalper_bot.strategies.signal_quality import canonical_max_spread_pct

    assert canonical_max_spread_pct() == resolve_entry_policy().execution_max_spread_pct

    monkeypatch.setenv("ORDER_MAX_SPREAD_PCT", "2.5")
    assert canonical_max_spread_pct() == 2.5


def test_candidate_selector_takes_its_floor_from_the_execution_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector

    monkeypatch.setenv("EXECUTION_MIN_OPTION_PREMIUM", "45")
    selector = TradeCandidateSelector()

    assert selector.min_option_premium == 45.0
