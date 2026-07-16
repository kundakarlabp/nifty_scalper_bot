from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from .assertions import assert_candle_ssot_consistent


@pytest.mark.e2e_live_sim
def test_full_live_trade_ce_target_exit(live_sim_system):
    s = live_sim_system
    scenario = s.scenario
    s.hydrate()
    assert len(s.history.record_requests()) == 4
    for symbol in s.candle_engines:
        assert_candle_ssot_consistent(
            symbol=symbol,
            candle_engine=s.candle_engines[symbol],
            indicator_engine=s.indicator_engine,
            runner=s.runner,
        )
    assert not s.broker.query_orders()

    s.subscribe_all()
    s.clock.advance_to(datetime(2026, 7, 15, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")))
    for minute in range(3):
        for symbol, price in scenario.live_ticks(s.clock.now()):
            s.exchange.publish_tick(
                symbol, ltp=price + minute, bid=price + minute - 0.2, ask=price + minute
            )
        s.clock.advance_to_next_minute()
    for symbol in s.candle_engines:
        s.exchange.publish_tick(
            symbol, ltp=101 if "CE" in symbol else 80 if "PE" in symbol else 25010
        )
        assert_candle_ssot_consistent(
            symbol=symbol,
            candle_engine=s.candle_engines[symbol],
            indicator_engine=s.indicator_engine,
            runner=s.runner,
        )

    s.evaluate_and_enter_ce(partial=True)
    assert s.broker.query_order(s.entry_order_id).average_price == 100
    assert s.broker.query_positions()[scenario.ce_symbol] == scenario.lot_size
    assert s.internal.active_stop[scenario.ce_symbol] == scenario.lot_size
    assert s.internal.active_target[scenario.ce_symbol] == scenario.lot_size
    s.trail_stop(100)
    s.trail_stop(104)
    s.close_via_target()

    s.event_recorder.assert_sequence(
        [
            "HISTORY_REQUESTED",
            "HISTORY_HYDRATED",
            "CANDLE_SSOT_READY",
            "SUBSCRIPTION_REQUESTED",
            "SUBSCRIPTION_CONFIRMED",
            "FIRST_CURRENT_GENERATION_TICK",
            "LIVE_ORDERS_ARMED",
            "STRATEGY_EVALUATED",
            "SIGNAL_GENERATED",
            "CANDIDATE_EXECUTION_READINESS",
            "RISK_APPROVED",
            "ENTRY_SUBMITTED",
            "ENTRY_ACKNOWLEDGED",
            "ENTRY_PARTIAL_FILL",
            "ENTRY_COMPLETE",
            "POSITION_OPENED",
            "STOP_SUBMITTED",
            "TARGET_SUBMITTED",
            "BRACKET_ACTIVE",
            "STOP_MODIFIED",
            "EXIT_COMPLETE",
            "SIBLING_CANCELLED",
            "POSITION_RECONCILED",
            "POSITION_FLAT",
            "TRADE_CLOSED",
            "PNL_FINALIZED",
        ]
    )
    assert s.broker.query_positions()[scenario.ce_symbol] == 0
    assert s.broker.realised_pnl[scenario.ce_symbol] == scenario.lot_size * 10
    assert s.exchange.pending_events == 0
    assert s.clock.pending_callbacks == 0
