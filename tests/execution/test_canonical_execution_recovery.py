from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import BracketManager
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.utils.errors import OrderPlacementError


SYMBOL = "NFO:NIFTY2662324050PE"
OTHER = "NFO:NIFTY2662324100CE"


def stop(manager: BracketManager) -> None:
    manager.shutdown()
    manager._watchdog_thread.join(timeout=1.0)


def test_production_facade_restores_bracket_automatically(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")
    first = BracketManager(order_manager=SimpleNamespace())
    stop(first)
    first.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        tp1_price=110.0,
        tp1_qty=30,
        activate_immediately=True,
    )
    bracket = first.get_bracket("entry-1")
    assert bracket is not None
    bracket._market_escalation_fired = True
    bracket._filled_exit_sync_order_id = "exit-1"
    bracket._filled_exit_sync_started_at = 123.0
    bracket.remaining_quantity = 35
    first._exit_rescue_attempts[bracket.bracket_id] = 1
    first.save_state()

    restored = BracketManager(order_manager=SimpleNamespace())
    stop(restored)
    recovered = restored.get_bracket("entry-1")
    assert recovered is not None
    assert recovered.remaining_quantity == 35
    assert recovered._market_escalation_fired is True
    assert recovered._filled_exit_sync_order_id == "exit-1"
    assert restored._exit_rescue_attempts[recovered.bracket_id] == 1
    assert restored._state_storage_path == str(tmp_path / "virtual_brackets.json")
    assert restored._state_storage_durable is False


def test_corrupt_snapshot_is_rejected_as_one_unit_and_freezes_entries(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")
    payload = {
        "schema_version": 2,
        "brackets": {
            "good": {
                "entry_order_id": "good",
                "symbol": SYMBOL,
                "side": "BUY",
                "quantity": 65,
                "remaining_quantity": 65,
                "entry_price": 100.0,
                "sl_trigger_price": 90.0,
                "tp_trigger_price": 120.0,
            },
            "bad": {"entry_order_id": "bad", "symbol": OTHER},
        },
    }
    (tmp_path / "virtual_brackets.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    manager = BracketManager(order_manager=SimpleNamespace())
    stop(manager)
    assert manager.get_bracket("good") is None
    assert manager.has_unresolved_exit() is True
    assert str(manager.get_first_unresolved_exit_bracket_id()).startswith(
        "persistence:"
    )


def test_unknown_broker_state_never_closes_production_bracket(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = SimpleNamespace(get_positions=lambda: None)
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    stop(manager)
    assert manager._position_flat_for_symbol(SYMBOL) is False
    assert manager._verify_position_closed(SYMBOL) is False
    assert manager._broker_position_quantity(SYMBOL) is None

    broker.get_positions = lambda: [{"symbol": SYMBOL}]
    assert manager._position_flat_for_symbol(SYMBOL) is False
    assert manager._broker_position_quantity(SYMBOL) is None


def test_valid_empty_net_never_falls_through_to_day(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = SimpleNamespace(
        get_positions=lambda: {
            "net": [],
            "day": [{"symbol": SYMBOL, "quantity": 65}],
        }
    )
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    stop(manager)
    assert manager._position_flat_for_symbol(SYMBOL) is True
    assert manager._broker_position_quantity(SYMBOL) == 0


def test_persist_failure_keeps_protection_and_freezes_new_entries(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")),
    )
    manager.confirm_entry_fill("entry-1", 101.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None and bracket.active and bracket.entry_confirmed
    assert manager.has_unresolved_exit() is True
    assert str(manager.get_first_unresolved_exit_bracket_id()).startswith(
        "persistence:"
    )


def test_live_mode_rejects_ephemeral_tmp_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")
    manager = BracketManager(
        order_manager=SimpleNamespace(is_live_mode=lambda: True)
    )
    stop(manager)
    assert manager.has_unresolved_exit() is True
    with pytest.raises(OSError):
        manager._get_storage_path()


def test_decoder_rejects_partial_rows_and_duplicate_symbols() -> None:
    with pytest.raises(PositionSnapshotError):
        decode_position_snapshot([{"symbol": SYMBOL}])
    with pytest.raises(PositionSnapshotError):
        decode_position_snapshot(
            [
                {"symbol": SYMBOL, "quantity": 65},
                {"symbol": SYMBOL, "quantity": 0},
            ]
        )


def test_local_loss_cannot_be_improved_by_delayed_broker_pnl(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.open_position(SYMBOL, "LONG", 65, 100.0)
    manager.close_position(SYMBOL, 90.0, "SL")
    local_loss = manager.get_realized_pnl()
    assert local_loss == pytest.approx(-650.0)
    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "realised": 0.0,
            }
        ]
    )
    assert manager.get_realized_pnl() == pytest.approx(local_loss)


class Hub:
    def get_quote(self, symbol: str, allow_pull: bool = True):
        return {"best_bid": 100.0, "best_ask": 101.0}


def test_zero_spread_limit_is_strict_for_options() -> None:
    with pytest.raises(OrderPlacementError):
        ExecutionPolicy(Hub(), max_spread_pct=0.0).build_plan(SYMBOL, "BUY")
