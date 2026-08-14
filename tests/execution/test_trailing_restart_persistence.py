from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager


def _manager() -> BracketManager:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    return manager


def test_confirmed_fill_reanchors_initial_risk_to_activated_stop() -> None:
    manager = _manager()
    manager.register_virtual_bracket(
        "fill-risk", "NFO:NIFTY2681824400PE", "BUY", 65, 114.10, 105.00, 132.35
    )

    manager.confirm_entry_fill("fill-risk", 113.55)

    bracket = manager.get_bracket("fill-risk")
    assert bracket is not None
    assert bracket.sl_trigger_price == 104.50
    assert bracket.initial_sl_trigger_price == bracket.sl_trigger_price


def test_fallback_trail_revision_survives_restart(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    symbol = "NFO:NIFTY2681824400PE"
    manager = _manager()
    manager.register_virtual_bracket(
        order_id="restart-trail",
        symbol=symbol,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=140.0,
        activate_immediately=False,
    )
    manager.confirm_entry_fill("restart-trail", 100.0)
    activated_at = manager.get_bracket("restart-trail").entry_fill_ts

    manager.on_tick(symbol, 110.0)
    trailed = manager.get_bracket("restart-trail")
    assert trailed is not None
    assert trailed.sl_trigger_price > 100.0
    assert trailed.highest_ltp == 110.0
    assert trailed.trail_revision == 1

    restarted_manager = _manager()
    assert restarted_manager.load_state() is True
    restored = restarted_manager.get_bracket("restart-trail")
    assert restored is not None
    assert restored.sl_trigger_price == trailed.sl_trigger_price
    assert restored.highest_ltp == trailed.highest_ltp
    assert restored.trail_revision == trailed.trail_revision
    assert restored.entry_fill_ts == activated_at


def test_time_stop_exemption_progress_survives_restart(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    symbol = "NFO:NIFTY2681824400PE"
    manager = _manager()
    manager.register_virtual_bracket(
        "restart-progress", symbol, "BUY", 65, 100.0, 90.0, 140.0
    )
    manager.confirm_entry_fill("restart-progress", 100.0)

    manager.on_tick(symbol, 105.0)

    restarted_manager = _manager()
    assert restarted_manager.load_state() is True
    restored = restarted_manager.get_bracket("restart-progress")
    assert restored is not None
    assert restored.highest_ltp == 105.0
    assert restored.trail_revision == 0


def test_sub_threshold_tick_does_not_write_bracket_snapshot(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    symbol = "NFO:NIFTY2681824400PE"
    manager = _manager()
    manager.register_virtual_bracket("no-trail", symbol, "BUY", 65, 100.0, 90.0, 140.0)
    manager.confirm_entry_fill("no-trail", 100.0)
    manager.save_state = Mock()

    manager.on_tick(symbol, 104.95)

    manager.save_state.assert_not_called()
