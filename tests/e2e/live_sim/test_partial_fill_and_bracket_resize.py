import pytest


@pytest.mark.e2e_live_sim
def test_partial_fill_and_bracket_resize(live_sim_system):
    s = live_sim_system
    s.hydrate()
    s.subscribe_all()
    s.evaluate_and_enter_ce(partial=True)
    symbol = s.scenario.ce_symbol
    s.broker.fill(
        s.entry_order_id,
        int(s.scenario.lot_size * 0.4),
        s.scenario.entry_price,
        duplicate=True,
    )
    assert s.internal.positions[symbol] == s.broker.query_positions()[symbol]
    assert s.internal.active_stop[symbol] == s.scenario.lot_size
    assert s.internal.active_target[symbol] == s.scenario.lot_size
    stop_id = s.bracket_orders[symbol]
    target_id = s.target_orders[symbol]
    s.broker.modify_order(stop_id, quantity=s.scenario.lot_size)
    s.broker.modify_order(target_id, quantity=s.scenario.lot_size)
    s.invariants.check_all()
    s.close_via_target()
    assert s.internal.positions[symbol] == 0
    assert s.broker.pending_callbacks == 0
