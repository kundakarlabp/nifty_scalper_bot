from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager


def _manager() -> BracketManager:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    return manager


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

    manager.on_tick(symbol, 110.0)
    trailed = manager.get_bracket("restart-trail")
    assert trailed is not None
    assert trailed.sl_trigger_price > 100.0
    assert trailed.highest_ltp == 110.0
    assert trailed.trail_revision == 1

    restored = _manager().get_bracket("restart-trail")
    assert restored is not None
    assert restored.sl_trigger_price == trailed.sl_trigger_price
    assert restored.highest_ltp == trailed.highest_ltp
    assert restored.trail_revision == trailed.trail_revision


def test_sub_threshold_tick_does_not_write_bracket_snapshot(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    symbol = "NFO:NIFTY2681824400PE"
    manager = _manager()
    manager.register_virtual_bracket("no-trail", symbol, "BUY", 65, 100.0, 90.0, 140.0)
    manager.confirm_entry_fill("no-trail", 100.0)
    manager.save_state = Mock()

    manager.on_tick(symbol, 105.0)

    manager.save_state.assert_not_called()
