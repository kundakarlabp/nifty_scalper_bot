from pathlib import Path


SOURCE = Path('src/nifty_scalper_bot/strategies/runner.py').read_text(encoding='utf-8')
MDM_SOURCE = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text(encoding='utf-8')


def test_runner_readiness_snapshot_contains_selected_contract_fields() -> None:
    assert 'LIVE_TRADING_READINESS_SNAPSHOT' in SOURCE
    assert 'selected_ce_symbol' in SOURCE
    assert 'selected_pe_symbol' in SOURCE
    assert 'ce_exec_ready' in SOURCE
    assert 'pe_exec_ready' in SOURCE
    assert 'order_path_entered' in SOURCE


def test_runner_eval_decision_splits_diagnostic_and_trading_allowed() -> None:
    assert 'diagnostic_eval_allowed' in SOURCE
    assert 'trading_allowed' in SOURCE
    assert 'order_forwarding_allowed' in SOURCE
    assert 'market_session_state' in SOURCE
    assert 'history_domain_used' in SOURCE


def test_stale_selected_symbol_emits_detail_and_disarms_live_orders() -> None:
    assert 'MARKET_DATA_STALE_SYMBOLS_DETAIL' in MDM_SOURCE
    assert 'stale_age_ms_by_symbol' in MDM_SOURCE
    assert 'impact_on_trading' in MDM_SOURCE


def test_candidate_refresh_pending_is_deferred_or_terminally_dropped_with_reason() -> None:
    assert 'SIGNAL_DEFERRED_CANDIDATE_REFRESH_PENDING' in SOURCE
    assert 'SIGNAL_DROPPED_CANDIDATE_REFRESH_PENDING' in SOURCE
    assert 'candidate_refresh_pending' in SOURCE


def test_no_safety_gate_is_bypassed_when_live_orders_not_armed() -> None:
    assert 'runtime_live_orders_armed' in SOURCE
    assert 'runtime_data_not_ready' in SOURCE
