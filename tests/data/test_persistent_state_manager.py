from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nifty_scalper_bot.data.persistent_state import PersistentStateManager


def test_persistent_state_roundtrip(tmp_path: Path) -> None:
    base_path = tmp_path / "state"
    base_path.mkdir()
    manager = PersistentStateManager(base_path=base_path)
    trade = {
        "symbol": "NIFTY25OCT20000CE",
        "action": "BUY",
        "quantity": 25,
        "price": 101.25,
        "status": "FILLED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manager.save_trade(trade)
    position = {
        "symbol": "NIFTY25OCT20000CE",
        "side": "LONG",
        "quantity": 25,
        "entry_price": 101.25,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "current_price": 102.0,
    }
    manager.save_position(position)
    fill = {
        "order_id": "OID123",
        "symbol": "NIFTY25OCT20000CE",
        "side": "BUY",
        "quantity": 25,
        "fill_price": 101.25,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manager.save_fill(fill)
    manager.save_shadow_equity(1500.0, {"NIFTY25OCT20000CE": 1500.0})
    manager.flush()
    manager.close()

    restored = PersistentStateManager(base_path=base_path)
    trades = restored.load_trades()
    assert trades
    assert trades[0]["symbol"] == "NIFTY25OCT20000CE"
    positions = restored.load_positions()
    assert any(item["symbol"] == "NIFTY25OCT20000CE" for item in positions)
    fills = restored.load_fills()
    assert fills
    assert fills[0]["order_id"] == "OID123"
    equity, exposures = restored.load_shadow_state()
    assert equity == pytest.approx(1500.0)
    assert exposures["NIFTY25OCT20000CE"] == pytest.approx(1500.0)
    restored.close()
