import pytest

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.live_runtime_e2e]


def test_context_candidate_side_isolation(live_sim_system):
    system = live_sim_system
    system.start()
    system.hydrate_via_production_path()
    system._publish_context_ticks()  # noqa: SLF001 - market-data-only setup
    ce_ready = system.evaluate_candidate_readiness(system.scenario.ce_symbol)
    assert ce_ready is not None and ce_ready.executable
    risk_before = system.observers.risk_requests
    pe_ready = system.evaluate_candidate_readiness(system.scenario.pe_symbol)
    assert pe_ready is not None
    assert system.observers.risk_requests == risk_before
    mismatch = system.evaluate_candidate_readiness("NFO:NIFTY26JUL25100CE")
    assert mismatch is not None and not mismatch.executable
    assert system.observers.risk_requests == risk_before
