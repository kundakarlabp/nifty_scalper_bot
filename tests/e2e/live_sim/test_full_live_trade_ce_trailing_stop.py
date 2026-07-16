import pytest

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.simulation_component]


def test_full_live_trade_ce_trailing_stop(live_sim_system):
    system = live_sim_system
    system.start()
    system.hydrate_via_production_path()
    system.publish_market_scenario(exit_mode="stop")
    system.run_until_flat()
    events = system.event_recorder
    events.assert_present("SIGNAL_GENERATED")
    events.assert_present("BRACKET_ACTIVE")
    events.assert_present("EXIT_COMPLETE")
    events.assert_present("POSITION_FLAT")
    stop_prices = [
        price
        for _event, payload in system.observers.bracket_events
        if (price := payload.get("sl")) is not None
    ]
    assert stop_prices == sorted(stop_prices)
