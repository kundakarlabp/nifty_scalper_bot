from __future__ import annotations

import threading
from types import SimpleNamespace

from nifty_scalper_bot.execution.bracket_core import BracketManager, BracketState
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager


def test_orphan_adoption_returns_existing_canonical_owner_without_repricing() -> None:
    manager = BracketManager.__new__(BracketManager)
    manager._lock = threading.RLock()
    manager._reconcile_lock = threading.Lock()
    manager._orphan_retry_last_attempt = {}
    manager._orphan_retry_count = {}
    owner = BracketState(
        entry_order_id="entry-1",
        symbol="NFO:NIFTY26SEP22850CE",
        side="BUY",
        quantity=65,
        entry_price=108.55,
        sl_trigger_price=105.20,
        tp_trigger_price=116.65,
        active=True,
        entry_confirmed=True,
    )
    manager._brackets = {"entry-1": owner}

    adopted = manager.attach_orphan_position("NFO:NIFTY26SEP22850CE", "BUY", 65, 108.55)

    assert adopted == "entry-1"
    assert list(manager._brackets) == ["entry-1"]
    assert owner.sl_trigger_price == 105.20
    assert owner.tp_trigger_price == 116.65


def test_completed_trade_mae_includes_executable_exit_fill() -> None:
    manager = LedgerBracketManager.__new__(LedgerBracketManager)
    manager._fill_ledger = None
    manager._broker_costs_for_confirmed_fills = lambda bracket, fills: None
    manager._trail_activation_r = lambda bracket: 0.5

    bracket = SimpleNamespace(
        bracket_id="entry-1",
        symbol="NFO:NIFTY26SEP22850CE",
        side="BUY",
        quantity=65,
        entry_fill_price=108.55,
        entry_price=108.55,
        highest_ltp=108.70,
        lowest_ltp=108.55,
        initial_sl_trigger_price=105.20,
        sl_trigger_price=105.20,
        closed_at=1002.0,
        entry_fill_ts=1000.0,
        created_at=999.0,
        trade_provenance={},
        trail_revision=0,
        exit_reason="WATCHDOG_HARD_SL",
        close_source="broker_fill",
        exit_arrival_price=104.95,
        exit_quote_bid=104.95,
        exit_quote_ask=105.05,
        exit_triggered_at=1001.0,
        exit_submitted_at=1001.2,
        exit_order_type="LIMIT",
        exit_market_fallback=False,
        exit_rejected_attempts=0,
    )

    outcome = manager._completed_trade_outcome(
        bracket,
        ledger_pnl=None,
        gross_pnl=-230.75,
        exit_price=105.00,
        ledger_complete=True,
    )

    assert outcome["mae_points"] == 3.55
    assert outcome["mae_r"] > 1.0
    assert outcome["mfe_points"] == 0.15
