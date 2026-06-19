from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def test_position_reconciliation_failure_blocks_live_arming_until_successful_recompute():
    failed = normalize_readiness_blockers(
        ["position_reconciliation_failed"],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert failed.live_orders_armed is False
    assert failed.primary_blocker == "position_reconciliation_failed"

    cleared = normalize_readiness_blockers(
        [],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert cleared.live_orders_armed is True


def test_unresolved_exit_position_blocks_live_arming():
    decision = normalize_readiness_blockers(
        ["unresolved_exit_position"],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert decision.live_orders_armed is False
    assert decision.primary_blocker == "unresolved_exit_position"
