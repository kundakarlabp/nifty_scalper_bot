from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def test_market_closed_dominates_quote_missing_with_secondary_debug() -> None:
    decision = normalize_readiness_blockers(
        ["selected_option_quote_missing"],
        "closed",
        live_mode=True,
        evaluation_ready=False,
        execution_ready=False,
    )

    assert decision.primary_blocker == "market_closed"
    assert decision.blocker_list == ["market_closed"]
    assert "selected_option_quote_missing" in decision.secondary_blockers
    assert decision.human_reason == "market_closed"


def test_market_open_quote_missing_is_primary() -> None:
    decision = normalize_readiness_blockers(
        ["selected_ce_quote_missing"],
        "open",
        live_mode=True,
        evaluation_ready=False,
        execution_ready=False,
    )

    assert decision.primary_blocker == "selected_option_quote_missing"
    assert decision.secondary_blockers == []


def test_emergency_stop_has_priority_over_market_closed() -> None:
    decision = normalize_readiness_blockers(
        ["selected_option_quote_missing"],
        "closed",
        emergency_state={"emergency_stop_active": True},
        live_mode=True,
    )

    assert decision.primary_blocker == "emergency_stop_active"


def test_broker_auth_invalid_has_priority_over_market_closed() -> None:
    decision = normalize_readiness_blockers(
        ["market_closed"],
        "closed",
        broker_state={"broker_auth_invalid": True},
        live_mode=True,
    )

    assert decision.primary_blocker == "broker_auth_invalid"
