import pytest


@pytest.mark.e2e_live_sim
def test_context_candidate_side_isolation(live_sim_system):
    s = live_sim_system
    ce = s.scenario.ce_symbol
    pe = s.scenario.pe_symbol
    s.hydrate()
    s.event_recorder.record("STRATEGY_EVALUATED", s.scenario.spot_symbol)
    s.event_recorder.record("SIGNAL_GENERATED", ce, side="CE")
    s.event_recorder.record(
        "CANDIDATE_EXECUTION_READINESS", ce, ready=True, opposite_side="stale"
    )
    s.event_recorder.record("RISK_APPROVED", ce)
    oid = s.broker.place_order(
        symbol=ce, side="BUY", quantity=75, order_type="LIMIT", price=100
    )
    s.event_recorder.record("ENTRY_SUBMITTED", ce, order_id=oid)
    assert len(s.broker.query_orders()) == 1

    s.event_recorder.record("STRATEGY_EVALUATED", s.scenario.spot_symbol)
    s.event_recorder.record("SIGNAL_GENERATED", pe, side="PE")
    s.event_recorder.record(
        "CANDIDATE_EXECUTION_READINESS",
        pe,
        ready=False,
        blocker_code="selected_pe_unready",
    )
    s.event_recorder.record(
        "CANDIDATE_EXECUTION_READINESS",
        "NFO:OTHERCE",
        ready=False,
        blocker_code="selected_contract_mismatch",
    )
    s.event_recorder.assert_count("RISK_APPROVED", 1)
    assert len(s.broker.query_orders()) == 1
