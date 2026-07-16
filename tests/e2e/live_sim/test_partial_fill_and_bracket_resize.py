import pytest

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.live_runtime_e2e]


def test_partial_fill_and_bracket_resize(live_sim_system):
    system = live_sim_system
    system.start()
    system.hydrate_via_production_path()
    system.publish_partial_fill_scenario()
    system.run_until_flat()
    events = system.event_recorder
    events.assert_present("ENTRY_PARTIAL_FILL")
    events.assert_present("ENTRY_COMPLETE")
    events.assert_present("BRACKET_ACTIVE")
    events.assert_present("POSITION_FLAT")
    assert system.broker.pending_callbacks == 0
