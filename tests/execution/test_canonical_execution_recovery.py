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
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NSB_TEST_MODE", raising=False)
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


def test_restart_with_already_protected_position_no_duplicate_bracket(
    tmp_path, monkeypatch
) -> None:
    """Slice-2(d): restart restores the protected bracket; a replayed
    registration for the same symbol must not create a second bracket."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")
    first = BracketManager(order_manager=SimpleNamespace())
    stop(first)
    first.register_virtual_bracket(
        order_id="entry-protected",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=145.15,
        sl=144.50,
        tp=152.00,
        activate_immediately=False,
    )
    first.confirm_entry_fill("entry-protected", 145.15)
    first.save_state()

    restored = BracketManager(order_manager=SimpleNamespace())
    stop(restored)
    recovered = restored.get_bracket("entry-protected")
    assert recovered is not None
    assert recovered.active and recovered.entry_confirmed
    assert recovered.remaining_quantity == 65
    assert len([b for b in restored._brackets.values() if b.symbol == SYMBOL]) == 1

    # Replayed registration (e.g. recovery logic re-running) must dedupe.
    restored.register_virtual_bracket(
        order_id="entry-protected",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=145.15,
        sl=144.50,
        tp=152.00,
        activate_immediately=False,
    )
    brackets = [b for b in restored._brackets.values() if b.symbol == SYMBOL]
    assert len(brackets) == 1
    assert brackets[0].remaining_quantity == 65
    assert brackets[0].active and brackets[0].entry_confirmed


def test_broker_exposure_states_distinguish_flat_absent_nonzero_and_unknown(tmp_path):
    from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState

    pm = PositionManager(str(tmp_path / "positions.json"))
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.UNKNOWN

    pm.synchronize_with_broker({"net": [{"tradingsymbol": SYMBOL.replace("NFO:", ""), "quantity": 0, "product": "MIS"}], "day": []})
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.FLAT
    assert pm.broker_exposure_state(OTHER) is BrokerExposureState.ABSENT

    pm.synchronize_with_broker([{"symbol": SYMBOL, "quantity": 65, "average_price": 100.0, "last_price": 100.0, "product": "MIS"}])
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.NONZERO

    with pytest.raises(PositionSnapshotError):
        pm.synchronize_with_broker([{"symbol": SYMBOL, "product": "MIS"}])
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.NONZERO


@pytest.mark.parametrize(
    "payload",
    [
        [{"symbol": SYMBOL, "quantity": "bad"}],
        [{"symbol": SYMBOL}],
        [{"symbol": SYMBOL, "quantity": 1}, {"tradingsymbol": SYMBOL.replace("NFO:", ""), "quantity": 0}],
    ],
)
def test_malformed_broker_exposure_never_decodes_as_flat(payload):
    with pytest.raises(PositionSnapshotError):
        decode_position_snapshot(payload)


def test_broker_exposure_snapshot_expires_to_unknown(tmp_path, monkeypatch):
    from nifty_scalper_bot.execution import position_manager as pm_module
    from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState

    now = {"value": 100.0}
    monkeypatch.setattr(pm_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setenv("BROKER_POSITION_SNAPSHOT_MAX_AGE_SECONDS", "20")
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.synchronize_with_broker([{"symbol": SYMBOL, "quantity": 0, "product": "MIS"}])

    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.FLAT
    snapshot = pm.broker_exposure_snapshot()
    assert snapshot["fresh"] is True
    assert snapshot["max_age_seconds"] == 20.0

    now["value"] = 121.0
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.UNKNOWN
    assert pm.broker_exposure_snapshot()["fresh"] is False


def test_absent_snapshot_expires_and_local_generation_invalidates(tmp_path, monkeypatch):
    from nifty_scalper_bot.execution import position_manager as pm_module
    from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState

    now = {"value": 200.0}
    monkeypatch.setattr(pm_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setenv("BROKER_POSITION_SNAPSHOT_MAX_AGE_SECONDS", "20")
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.synchronize_with_broker([])
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.ABSENT

    pm.open_position(SYMBOL, "LONG", 65, 100.0, order_id="manual")
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.UNKNOWN

    pm.synchronize_with_broker([{"symbol": SYMBOL, "quantity": 65, "average_price": 100.0, "last_price": 100.0, "product": "MIS"}])
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.NONZERO
    pm.update_position_price(SYMBOL, 101.0)
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.NONZERO

    now["value"] = 221.0
    assert pm.broker_exposure_state(SYMBOL) is BrokerExposureState.UNKNOWN


def test_position_manager_is_flat_fails_closed_on_lookup_exception(tmp_path, caplog):
    class RaisingPositions(dict):
        def get(self, _key, _default=None):
            raise RuntimeError("lookup failed")

    errors = []
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm._positions = RaisingPositions()
    pm._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        error=lambda *args, **kwargs: errors.append((args, kwargs)),
    )

    assert pm.is_flat(SYMBOL) is False
    assert any("POSITION_FLAT_CHECK_FAILED" in str(args[0]) for args, _kwargs in errors)
