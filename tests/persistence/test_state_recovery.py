from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nifty_scalper_bot.data.persistent_state import PersistentStateManager
from nifty_scalper_bot.execution.position_manager import PositionManager


def test_position_manager_recovers_from_persistent_store(tmp_path: Path) -> None:
    base_path = tmp_path / "store"
    manager = PersistentStateManager(base_path=base_path)

    symbol = "NIFTY24NOV18000CE"
    timestamp = datetime.now(timezone.utc).isoformat()
    manager.save_trade(
        {
            "symbol": symbol,
            "action": "BUY",
            "quantity": 25,
            "price": 101.25,
            "status": "FILLED",
            "timestamp": timestamp,
        }
    )
    manager.save_position(
        {
            "symbol": symbol,
            "side": "LONG",
            "quantity": 25,
            "entry_price": 101.25,
            "entry_time": timestamp,
            "current_price": 101.25,
        }
    )
    manager.save_order(
        {
            "order_id": "OPEN-123",
            "symbol": symbol,
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 25,
            "price": 101.25,
            "status": "OPEN",
            "timestamp": timestamp,
            "filled_quantity": 0,
        }
    )
    manager.save_shadow_equity(2500.0, {symbol: 2500.0})
    manager.flush()
    manager.close()

    restored_manager = PersistentStateManager(base_path=base_path)

    state_path = tmp_path / "positions_state.json"
    position_manager = PositionManager(state_file=str(state_path))
    position_manager.attach_persistent_state(restored_manager)
    position_manager.load_state()

    positions = position_manager.get_open_positions()
    assert len(positions) == 1
    assert positions[0].symbol == symbol

    pending_orders = position_manager.get_pending_orders()
    assert any(order.order_id == "OPEN-123" for order in pending_orders)

    trades = restored_manager.load_trades()
    assert trades
    assert trades[0]["symbol"] == symbol

    equity, exposures = restored_manager.load_shadow_state()
    assert equity == pytest.approx(2500.0)
    assert exposures[symbol] == pytest.approx(2500.0)

    restored_manager.close()
