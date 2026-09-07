from pathlib import Path


def test_runner_arms_existing_final_snapshot_freshness_contract() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert '"LIVE_ENTRY_MAX_SIGNAL_AGE_SECONDS", 5.0, minimum=0.1' in source
    assert '"LIVE_ENTRY_MAX_ADVERSE_DRIFT_PCT", 1.0, minimum=0.05' in source
    assert '"decision_ts": decision_ts' in source
    assert '"decision_reference_price": float(price)' in source


def test_freshness_is_armed_on_entry_plan_not_as_new_execution_path() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    block_start = source.index("decision_ts = float(timestamp.timestamp())")
    block_end = source.index("submit_result = self._order_manager.submit_trade_plan_result(plan)", block_start)
    block = source[block_start:block_end]
    assert 'intent="ENTRY"' in block
    assert "max_signal_age_seconds=" in block
    assert "max_entry_drift_pct=" in block
    assert ".place_order(" not in block
