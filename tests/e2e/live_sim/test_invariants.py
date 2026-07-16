import pytest

from .invariants import InternalState, TradingInvariantChecker
from .simulated_broker import SimulatedBroker
from .virtual_clock import VirtualClock


def test_invariant_checker_reports_precise_failures():
    broker = SimulatedBroker(VirtualClock())
    state = InternalState(positions={"S": 1})
    checker = TradingInvariantChecker(broker, state)
    with pytest.raises(AssertionError, match="unprotected open position"):
        checker.check_all()
    oid = broker.place_order(symbol="S", side="BUY", quantity=1, order_type="MARKET")
    broker.fill(oid, 1, 10)
    state.positions["S"] = 2
    state.active_stop["S"] = 2
    with pytest.raises(AssertionError, match="broker/internal quantity mismatch"):
        checker.check_all()
    state.positions["S"] = 1
    state.active_stop["S"] = 2
    with pytest.raises(AssertionError, match="stop quantity greater"):
        checker.check_all()
    state.active_stop["S"] = 1
    state.stop_prices["S"] = [9, 8]
    with pytest.raises(AssertionError, match="stop loosening"):
        checker.check_all()
    state.stop_prices["S"] = [9, 10]
    broker.positions["S"] = 0
    state.positions["S"] = 0
    state.active_stop["S"] = 0
    state.active_target["S"] = 1
    with pytest.raises(AssertionError, match="active bracket after flat"):
        checker.check_all()
