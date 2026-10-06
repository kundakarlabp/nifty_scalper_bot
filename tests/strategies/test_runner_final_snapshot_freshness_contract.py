from pathlib import Path


def test_runner_arms_final_trade_plan_freshness_contract() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert '"ENTRY_MAX_SIGNAL_AGE_SECONDS", "15"' in source
    assert '"ENTRY_MAX_PRICE_DRIFT_PCT", "2.0"' in source
    assert '"decision_ts": now_epoch' in source
    assert '"entry_submit_ts": _entry_submit_ts' in source


def test_freshness_is_armed_on_canonical_entry_plan() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    block_start = source.index("plan = TradePlan(")
    block_end = source.index(
        "submit_result = submit_result_fn(plan)", block_start
    )
    block = source[block_start:block_end]
    assert 'max_signal_age_seconds=max(' in block
    assert 'max_entry_drift_pct=max(' in block
    assert '"decision_ts": now_epoch' in block
    assert ".place_order(" not in block
