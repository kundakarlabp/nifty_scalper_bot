import pytest

from .event_recorder import EventRecorder
from .virtual_clock import VirtualClock

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.simulation_component]


def test_event_recorder_assertions_and_diagnostics():
    events = EventRecorder(VirtualClock())
    events.record("A")
    events.record("B", "SYM")
    events.assert_present("A")
    events.assert_exactly_once("B")
    events.assert_before("A", "B")
    events.assert_sequence(["A", "B"])
    assert events.filter(event="B", symbol="SYM")
    with pytest.raises(AssertionError, match="count"):
        events.assert_count("A", 2)
