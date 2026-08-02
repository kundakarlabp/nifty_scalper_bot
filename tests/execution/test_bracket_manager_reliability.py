"""Reliability tests for virtual bracket exit execution."""

from __future__ import annotations

import logging
import time
from typing import Mapping
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _build_manager() -> tuple[BracketManager, Mock]:
    """Create manager with mock dependencies. Args: none; Returns: tuple; Raises: none."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.wait_for_fill.return_value = True
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda symbol, qty: f"exit-{symbol}-{qty}")
    return manager, order_manager


def _wait_for_exit(manager: BracketManager, entry_id: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    bracket = manager.get_bracket(entry_id)
    while time.time() < deadline and bracket is not None and not bracket.exit_executed:
        time.sleep(0.01)
        bracket = manager.get_bracket(entry_id)
    return bracket


def test_stop_loss_cross_detection_on_jump() -> None:
    """Ensure SL triggers on jump across threshold."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol="NFO:NIFTYTEST",
        side="BUY",
        qty=1,
        price=105.0,
        sl=100.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-1", 105.0)

    manager.on_tick("NFO:NIFTYTEST", 105.0)
    manager.on_tick("NFO:NIFTYTEST", 104.0)
    manager.on_tick("NFO:NIFTYTEST", 97.0)

    bracket = _wait_for_exit(manager, "entry-1")
    assert bracket is not None
    assert bracket.exit_executed is True
    assert bracket.active is False


def test_trailing_stop_never_moves_backwards() -> None:
    """Ensure manual trailing updates are monotonic."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-2",
        symbol="NFO:NIFTYTEST2",
        side="BUY",
        qty=1,
        price=100.0,
        sl=95.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-2", 100.0)
    manager.update_trailing_sl("NFO:NIFTYTEST2", 98.0)
    manager.update_trailing_sl("NFO:NIFTYTEST2", 96.0)
    bracket = manager.get_bracket("entry-2")
    assert bracket is not None
    assert bracket.sl_trigger_price == 98.0


def test_fallback_market_exit_executes_after_retry_exhaustion() -> None:
    """Ensure fallback market exit runs if retries fail."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.place_order.return_value = "fallback-1"
    order_manager.wait_for_fill.return_value = True
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda _symbol, _qty: None)
    manager.register_virtual_bracket(
        order_id="entry-3",
        symbol="NFO:NIFTYTEST3",
        side="BUY",
        qty=1,
        price=100.0,
        sl=99.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-3", 100.0)

    manager.on_tick("NFO:NIFTYTEST3", 98.0)

    deadline = time.time() + 1.0
    while time.time() < deadline and not order_manager.place_order.called:
        time.sleep(0.01)

    assert order_manager.place_order.called
    bracket = _wait_for_exit(
        manager, "entry-3", timeout_s=max(0.0, deadline - time.time())
    )
    assert bracket is not None
    assert bracket.exit_executed is True


def test_exit_state_mutates_only_after_broker_confirmation() -> None:
    """Ensure state mutates only after broker confirmation."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.wait_for_fill.return_value = False
    order_manager.place_order.return_value = None
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda _symbol, _qty: "exit-pending")
    manager.register_virtual_bracket(
        order_id="entry-4",
        symbol="NFO:NIFTYTEST4",
        side="BUY",
        qty=1,
        price=100.0,
        sl=99.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-4", 100.0)

    manager.on_tick("NFO:NIFTYTEST4", 98.0)
    bracket = manager.get_bracket("entry-4")
    assert bracket is not None
    assert bracket.exit_executed is False
    assert bracket.active is True


def test_websocket_tick_jump_scenario_triggers_sl() -> None:
    """Ensure websocket jump sequence triggers SL."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-5",
        symbol="NFO:NIFTYTEST5",
        side="BUY",
        qty=1,
        price=105.0,
        sl=100.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-5", 105.0)

    for tick in [105.0, 104.0, 97.0]:
        manager.on_tick_event({"symbol": "NFO:NIFTYTEST5", "ltp": tick})

    bracket = _wait_for_exit(manager, "entry-5")
    assert bracket is not None
    assert bracket.exit_executed is True


def test_trail_update_does_not_exit_same_tick_regression() -> None:
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-1"
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        order_id="incident-1",
        symbol="NFO:NIFTY24100CE",
        side="BUY",
        qty=65,
        price=70.80,
        sl=66.60,
        tp=78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("incident-1")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = 0.20

    # Breakeven is cost-adjusted: exiting at the entry premium is a loss of the
    # full round-trip charge, so the lock sits at entry + cost_per_unit (~0.89).
    manager.on_tick("NFO:NIFTY24100CE", 72.60, exchange_ts=1_000.0)

    assert bracket.sl_trigger_price == 71.70
    assert bracket.exit_pending is False
    assert order_manager.place_order.call_count == 0

    manager.on_tick("NFO:NIFTY24100CE", 72.30, exchange_ts=1_001.0)
    assert bracket.exit_pending is False
    assert order_manager.place_order.call_count == 0

    manager.on_tick("NFO:NIFTY24100CE", 71.65, exchange_ts=1_002.0)
    assert "SL" in str(bracket.exit_reason or "")
    assert order_manager.place_order.call_count == 1


def test_duplicate_same_tick_has_one_trail_update_and_no_exit() -> None:
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-dup"
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "dup-1",
        "NFO:NIFTY24100CE",
        "BUY",
        65,
        70.80,
        66.60,
        78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("dup-1")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = 0.20

    manager.on_tick("NFO:NIFTY24100CE", 72.60, exchange_ts=2_000.0)
    manager.on_tick("NFO:NIFTY24100CE", 72.60, exchange_ts=2_000.0)

    assert bracket.sl_trigger_price == 71.70
    assert bracket.trail_revision == 1
    assert order_manager.place_order.call_count == 0


def test_one_lot_does_not_allocate_sub_lot_tp1_but_two_lot_keeps_whole_lot_tp1() -> (
    None
):
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        "one-lot",
        "NFO:NIFTYONECE",
        "BUY",
        65,
        100.0,
        95.0,
        120.0,
        tp1_price=110.0,
        tp1_qty=25,
    )
    one = manager.get_bracket("one-lot")
    assert one is not None
    assert one.tp_levels == []

    manager.register_virtual_bracket(
        "two-lot",
        "NFO:NIFTYTWOCE",
        "BUY",
        130,
        100.0,
        95.0,
        120.0,
        tp1_price=110.0,
        tp1_qty=65,
    )
    two = manager.get_bracket("two-lot")
    assert two is not None
    assert [(t.name, t.quantity) for t in two.tp_levels] == [("TP1", 65)]


def test_selection_drift_does_not_change_existing_bracket() -> None:
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        "open-ce", "NFO:NIFTY24100CE", "BUY", 65, 100.0, 95.0, 120.0
    )
    manager.confirm_entry_fill("open-ce", 100.0)
    manager.register_virtual_bracket(
        "new-ce", "NFO:NIFTY24050CE", "BUY", 65, 100.0, 95.0, 120.0
    )

    existing = manager.get_bracket("open-ce")
    assert existing is not None
    assert existing.symbol == "NFO:NIFTY24100CE"
    assert existing.active is True
    assert manager.get_bracket("new-ce") is not None


def test_duplicate_callback_submits_one_exit_order_only() -> None:
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-once"
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "exit-once",
        "NFO:NIFTYEXITCE",
        "BUY",
        65,
        100.0,
        95.0,
        120.0,
        activate_immediately=True,
    )

    manager.on_tick("NFO:NIFTYEXITCE", 94.0, exchange_ts=3_000.0)
    manager.on_tick("NFO:NIFTYEXITCE", 94.0, exchange_ts=3_000.0)

    assert order_manager.place_order.call_count == 1


def test_breakeven_activation_requires_configured_r_threshold() -> None:
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        "be-1",
        "NFO:NIFTYBECE",
        "BUY",
        65,
        70.80,
        66.60,
        78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("be-1")
    assert bracket is not None

    manager.on_tick("NFO:NIFTYBECE", 71.65, exchange_ts=4_000.0)
    assert bracket.sl_trigger_price == 66.60

    manager.on_tick("NFO:NIFTYBECE", 73.95, exchange_ts=4_001.0)
    assert bracket.sl_trigger_price >= 70.80
    assert bracket.sl_trigger_price < 73.95
    assert bracket.exit_pending is False


def test_adaptive_controller_trail_callback_uses_same_r_guard() -> None:
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        "adaptive-r",
        "NFO:NIFTYADAPTCE",
        "BUY",
        65,
        70.80,
        66.60,
        78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("adaptive-r")
    assert bracket is not None
    bracket.last_ltp = 71.65
    bracket.highest_ltp = 71.65

    assert manager._virtual_modify_sl(bracket.virtual_sl_id, 70.80) is False
    assert bracket.sl_trigger_price == 66.60
    assert bracket.trail_revision == 0

    bracket.last_ltp = 73.95
    bracket.highest_ltp = 73.95
    assert manager._virtual_modify_sl(bracket.virtual_sl_id, 70.80) is True
    assert bracket.sl_trigger_price == 70.80
    assert bracket.trail_revision == 1
    assert bracket.exit_pending is False


def test_timestampless_duplicate_callback_has_one_trail_revision_and_notification() -> (
    None
):
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-1"
    manager = BracketManager(order_manager=order_manager)
    notifications: list[tuple[str, Mapping[str, object] | None]] = []
    manager.set_notifier(lambda event, payload: notifications.append((event, payload)))
    manager.register_virtual_bracket(
        "no-ts-trail",
        "NFO:NIFTYNOTSCE",
        "BUY",
        65,
        70.80,
        66.60,
        78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("no-ts-trail")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = 0.20

    manager.process_exit_checks("NFO:NIFTYNOTSCE", 72.60)
    manager.process_exit_checks("NFO:NIFTYNOTSCE", 72.60)

    assert bracket.sl_trigger_price == 71.70
    assert bracket.trail_revision == 1
    assert [event for event, _ in notifications].count("BRACKET_TRAIL_UPDATED") <= 1
    assert order_manager.place_order.call_count == 0


def test_authoritative_resolved_lot_size_controls_tp1_allocation() -> None:
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        "one-custom-lot",
        "NFO:NIFTYCUSTOM1CE",
        "BUY",
        75,
        100.0,
        95.0,
        120.0,
        tp1_price=110.0,
        tp1_qty=25,
        resolved_lot_size=75,
    )
    one = manager.get_bracket("one-custom-lot")
    assert one is not None
    assert one.tp_levels == []

    manager.register_virtual_bracket(
        "two-custom-lot",
        "NFO:NIFTYCUSTOM2CE",
        "BUY",
        150,
        100.0,
        95.0,
        120.0,
        tp1_price=110.0,
        tp1_qty=75,
        resolved_lot_size=75,
    )
    two = manager.get_bracket("two-custom-lot")
    assert two is not None
    assert [(t.name, t.quantity) for t in two.tp_levels] == [("TP1", 75)]


def test_active_selection_drift_path_leaves_open_bracket_untouched(caplog) -> None:
    manager, order_manager = _build_manager()
    manager.register_virtual_bracket(
        "open-ce", "NFO:NIFTY24100CE", "BUY", 65, 100.0, 95.0, 120.0
    )
    manager.confirm_entry_fill("open-ce", 100.0)
    bracket = manager.get_bracket("open-ce")
    assert bracket is not None

    runner = object.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test.selection.drift")
    runner._active_selected_ce = "NFO:NIFTY24100CE"
    runner._active_selected_pe = "NFO:NIFTY24100PE"
    runner._active_selection_drift_log_key = None
    runner._active_selection_sync_log_key = None
    runner._active_option_symbols = {"NFO:NIFTY24100CE", "NFO:NIFTY24100PE"}
    runner._active_basket_all_symbols = set()
    runner._active_basket_token_by_symbol = {}
    runner._active_futures_symbol = None
    runner._active_symbols = set()
    runner._latest_context_snapshots = {}
    runner._sync_context_history_if_cold = lambda **_kwargs: None

    with caplog.at_level(logging.WARNING, logger="test.selection.drift"):
        runner.set_active_trading_universe(
            {
                "selected_ce": "NFO:NIFTY24050CE",
                "selected_pe": "NFO:NIFTY24050PE",
                "option_symbols": ["NFO:NIFTY24050CE", "NFO:NIFTY24050PE"],
                "symbols": ["NFO:NIFTY24050CE", "NFO:NIFTY24050PE"],
                "all_symbols": ["NFO:NIFTY24050CE", "NFO:NIFTY24050PE"],
                "basket_version": "drift-1",
            }
        )

    assert runner._active_selection_drift_log_key is not None
    assert runner._active_selected_ce == "NFO:NIFTY24050CE"
    assert bracket.symbol == "NFO:NIFTY24100CE"
    assert bracket.active is True
    assert bracket.sl_trigger_price == 95.0
    assert bracket.tp_trigger_price == 120.0
    assert bracket.remaining_quantity == 65
    assert order_manager.place_order.call_count == 0


def test_time_stop_flattens_a_stalled_position(monkeypatch) -> None:
    """A long option that reaches neither stop nor target still bleeds theta,
    so a stalled trade must be flattened once past the time stop."""
    import time as _time

    order_manager = Mock()
    order_manager.place_order.return_value = "exit-ts"
    monkeypatch.setenv("LIFECYCLE_TIME_STOP_MIN", "12")
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "stall-1",
        "NFO:NIFTYSTALLCE",
        "BUY",
        65,
        100.0,
        95.0,
        115.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("stall-1")
    assert bracket is not None
    bracket.entry_fill_ts = _time.time() - 800.0  # held > 12 min

    action = manager._time_stop_action(bracket, 101.0)
    assert action is not None
    assert action["type"] == "TIME_STOP"
    assert action["qty"] == bracket.remaining_quantity


def test_time_stop_spares_a_position_that_made_progress() -> None:
    import time as _time

    order_manager = Mock()
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "moving-1",
        "NFO:NIFTYMOVECE",
        "BUY",
        65,
        100.0,
        95.0,
        115.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("moving-1")
    assert bracket is not None
    bracket.entry_fill_ts = _time.time() - 800.0
    bracket.highest_ltp = 104.0  # 0.8R of progress, above the 0.5R floor

    assert manager._time_stop_action(bracket, 103.0) is None


def test_time_stop_is_inert_before_the_holding_window() -> None:
    import time as _time

    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "fresh-1",
        "NFO:NIFTYFRESHCE",
        "BUY",
        65,
        100.0,
        95.0,
        115.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("fresh-1")
    assert bracket is not None
    bracket.entry_fill_ts = _time.time() - 10.0

    assert manager._time_stop_action(bracket, 101.0) is None


def test_breakeven_lock_clears_round_trip_cost() -> None:
    """Locking at the raw entry premium books a guaranteed loss of the full
    round-trip charge; the lock must sit above it."""
    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "be-1",
        "NFO:NIFTYBECE",
        "BUY",
        65,
        70.80,
        66.60,
        78.25,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("be-1")
    assert bracket is not None
    bracket.trailing_config["breakeven_activation_r"] = 0.20

    cost = manager._breakeven_cost_per_unit(bracket)
    assert cost > 0.0

    manager.on_tick("NFO:NIFTYBECE", 72.60)
    assert bracket.sl_trigger_price > 70.80
    assert bracket.sl_trigger_price >= 70.80 + cost - 0.05


def test_trailing_tiers_are_keyed_to_r_not_premium_percent() -> None:
    """Identical premium moves on different stop distances must not receive
    identical protection — the tier ladder is denominated in R."""
    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "wide-1", "NFO:NIFTYWIDECE", "BUY", 65, 100.0, 80.0, 160.0,
        activate_immediately=True,
    )
    manager.register_virtual_bracket(
        "tight-1", "NFO:NIFTYTIGHTCE", "BUY", 65, 100.0, 98.0, 106.0,
        activate_immediately=True,
    )
    wide = manager.get_bracket("wide-1")
    tight = manager.get_bracket("tight-1")
    assert wide is not None and tight is not None

    # Same +4% premium move: 0.2R on the wide stop, 2.0R on the tight one.
    wide.highest_ltp = tight.highest_ltp = 104.0
    wide_sl = manager._calculate_tiered_trailing_sl(
        bracket=wide, ltp=104.0, profit_pct=4.0, high_water=104.0, atr=2.0
    )
    tight_sl = manager._calculate_tiered_trailing_sl(
        bracket=tight, ltp=104.0, profit_pct=4.0, high_water=104.0, atr=2.0
    )
    assert wide_sl is None, "0.2R must not trigger any protection tier"
    assert tight_sl is not None and tight_sl > 100.0


def test_completed_trade_outcome_reports_r_normalised_excursions() -> None:
    """Premium-point MAE/MFE cannot be compared across trades with different
    stop distances, so the outcome record carries R-normalised values."""
    from types import SimpleNamespace

    manager = BracketManager(order_manager=Mock())
    bracket = SimpleNamespace(
        bracket_id="r-1",
        entry_order_id="r-1",
        trade_lifecycle_id="r-1",
        symbol="NFO:NIFTYRCE",
        side="BUY",
        quantity=65,
        entry_price=100.0,
        entry_fill_price=100.0,
        initial_sl_trigger_price=96.0,
        sl_trigger_price=96.0,
        highest_ltp=108.0,
        lowest_ltp=98.0,
        close_source="broker_fill",
        exit_reason="TP",
        trade_provenance={},
        closed_at=None,
        entry_fill_ts=None,
        created_at=None,
        tag="",
    )

    outcome = manager._completed_trade_outcome(
        bracket,
        ledger_pnl=None,
        gross_pnl=520.0,
        exit_price=108.0,
        ledger_complete=True,
    )

    assert outcome["initial_risk_points"] == 4.0
    assert outcome["mfe_r"] == 2.0
    assert outcome["mae_r"] == 0.5
    assert outcome["r_multiple"] is not None
    assert outcome["r_multiple"] < 2.0  # net of costs


def _quote_tick(symbol: str, ltp: float, bid: float, ask: float) -> dict:
    return {
        "symbol": symbol,
        "last_price": ltp,
        "depth": {"buy": [{"price": bid, "quantity": 500}],
                  "sell": [{"price": ask, "quantity": 500}]},
        "exchange_timestamp": None,
    }


def test_long_stop_fires_when_bid_crosses_though_ltp_has_not() -> None:
    """A long option is sold into the bid. If the bid is already below the stop
    the position is executable below it, and waiting for LTP is late protection."""
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-bid"
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "bid-1", "NFO:NIFTYBIDCE", "BUY", 65, 100.0, 99.0, 120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("bid-1")
    assert bracket is not None

    manager.on_tick_event(_quote_tick("NFO:NIFTYBIDCE", 100.50, 98.50, 101.50))

    assert "SL" in str(bracket.exit_reason or "").upper()
    assert "src=bid" in str(bracket.exit_reason or "")
    assert order_manager.place_order.call_count >= 1


def test_short_stop_fires_when_ask_crosses_though_ltp_has_not() -> None:
    order_manager = Mock()
    order_manager.place_order.return_value = "exit-ask"
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        "ask-1", "NFO:NIFTYASKCE", "SELL", 65, 100.0, 101.0, 80.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("ask-1")
    assert bracket is not None

    manager.on_tick_event(_quote_tick("NFO:NIFTYASKCE", 99.50, 98.50, 101.50))

    assert "src=ask" in str(bracket.exit_reason or "")
    assert order_manager.place_order.call_count >= 1


def test_stop_holds_when_executable_side_has_not_crossed() -> None:
    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "hold-1", "NFO:NIFTYHOLDCE", "BUY", 65, 100.0, 97.0, 120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("hold-1")
    assert bracket is not None

    manager.on_tick_event(_quote_tick("NFO:NIFTYHOLDCE", 99.00, 98.50, 99.50))

    assert bracket.exit_pending is False


def test_stale_quote_falls_back_to_ltp_and_cannot_suppress_protection() -> None:
    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "stale-1", "NFO:NIFTYSTALECE", "BUY", 65, 100.0, 99.0, 120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("stale-1")
    assert bracket is not None
    # A generous bid captured long ago must not keep a breached position alive.
    manager._exit_quotes["NFO:NIFTYSTALECE"] = (105.0, 106.0, time.time() - 600.0)

    price, source = manager._executable_exit_price(bracket, 98.0)
    assert price == 98.0
    assert source == "ltp_stale_quote"


def test_tier4_uses_r_not_premium_percent() -> None:
    """A tight-stop trade at >3R but only ~2% premium profit must still trail;
    the Tier-4 guard previously compared premium-percent to an R threshold and
    returned no stop at all for the largest winners."""
    manager = BracketManager(order_manager=Mock())
    manager.register_virtual_bracket(
        "t4-1", "NFO:NIFTYT4CE", "BUY", 65, 100.0, 99.0, 130.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("t4-1")
    assert bracket is not None
    bracket.highest_ltp = 104.0  # 4R on a 1.0-point stop, only +4% premium

    new_sl = manager._calculate_tiered_trailing_sl(
        bracket=bracket, ltp=104.0, profit_pct=4.0, high_water=104.0, atr=0.0
    )
    assert new_sl is not None, "Tier 4 must engage on R, not premium percent"
    assert new_sl > 100.0
