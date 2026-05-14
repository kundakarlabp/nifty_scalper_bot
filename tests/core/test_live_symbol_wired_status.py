
def test_live_symbol_wired_success_logic_contract():
    mdm_tracked = True
    broker_ws_token_requested = False
    runner_callback_registered = False
    reason = 'dynamic_add'
    market_closed = False
    success = bool(
        mdm_tracked
        and (runner_callback_registered or reason in {'deferred_until_ready', 'history_pending'})
        and (broker_ws_token_requested or market_closed or reason in {'startup', 'history_pending'})
    )
    partial = (not success) and mdm_tracked
    assert success is False
    assert partial is True
