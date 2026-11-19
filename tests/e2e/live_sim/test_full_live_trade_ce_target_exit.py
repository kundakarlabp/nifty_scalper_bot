import pytest

from .assertions import assert_candle_ssot_consistent

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.simulation_component]


def test_full_live_trade_ce_target_exit(live_sim_system):
    system = live_sim_system
    system.start()
    system.hydrate_via_production_path()
    assert len(system.history.record_requests()) == 4
    for symbol in system.exchange.instruments:
        engine = system.runner._candle_engines[symbol]  # noqa: SLF001
        assert_candle_ssot_consistent(
            symbol=symbol,
            candle_engine=engine,
            indicator_engine=system.indicator_engine,
            runner=system.runner,
        )

    system.publish_market_scenario(exit_mode="target")
    system.run_until_flat()

    events = system.event_recorder
    events.assert_present("SIGNAL_GENERATED")
    events.assert_present("CANDIDATE_EXECUTION_READINESS")
    events.assert_present("ENTRY_COMPLETE")
    events.assert_present("BRACKET_ACTIVE")
    events.assert_present("EXIT_COMPLETE")
    events.assert_present("POSITION_FLAT")
    events.assert_present("PNL_FINALIZED")
    # Ordered audit across the existing live-simulation owners, without
    # bypassing risk, broker acknowledgements, or protective-exit handling.
    events.assert_sequence(
        [
            "LIVE_ORDERS_ARMED",
            "CANDLE_SSOT_READY",
            "SIGNAL_GENERATED",
            "CANDIDATE_SELECTED",
            "CANDIDATE_EXECUTION_READINESS",
            "RISK_APPROVED",
            "ENTRY_COMPLETE",
            "EXIT_COMPLETE",
            "POSITION_RECONCILED",
            "POSITION_FLAT",
            "PNL_FINALIZED",
        ]
    )
    assert (
        events.last("BRACKET_ACTIVE").sequence
        < events.last("EXIT_COMPLETE").sequence
    )
    assert events.last("CANDIDATE_EXECUTION_READINESS").payload["executable"]
    events.assert_absent("RISK_REJECTED")
    assert system.observers.broker_updates
    assert system.observers.signals
    assert system.observers.readiness_snapshots >= 1
    assert system.observers.risk_requests == 1
    assert system.observers.risk_approved == 1
    assert system.entry_order_id is not None
    assert system.broker.query_positions().get(system.scenario.ce_symbol, 0) == 0
    assert system.exchange.pending_events == 0
    assert system.clock.pending_callbacks == 0
