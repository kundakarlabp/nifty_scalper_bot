import pytest


@pytest.mark.e2e_live_sim
def test_full_live_trade_ce_trailing_stop(live_sim_system):
    s = live_sim_system
    s.hydrate()
    s.subscribe_all()
    s.evaluate_and_enter_ce()
    s.trail_stop(100)
    s.trail_stop(104)
    s.trail_stop(108)
    s.close_via_stop(108)
    prices = s.internal.stop_prices[s.scenario.ce_symbol]
    assert prices == sorted(prices)
    s.event_recorder.assert_sequence(
        [
            "STOP_SUBMITTED",
            "TARGET_SUBMITTED",
            "BRACKET_ACTIVE",
            "STOP_MODIFIED",
            "EXIT_COMPLETE",
            "SIBLING_CANCELLED",
            "POSITION_FLAT",
            "PNL_FINALIZED",
        ]
    )
    assert s.broker.realised_pnl[s.scenario.ce_symbol] == s.scenario.lot_size * 8
