from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.core.app import _reconcile_state
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.options.strike_selector import SelectedContract


def _sample_contract() -> SelectedContract:
    return SelectedContract(
        symbol="NIFTY25O2025450CE",
        option_type="CE",
        strike=25450.0,
        expiry=datetime(2025, 10, 30, tzinfo=timezone.utc),
        ltp=125.0,
        delta=None,
        metadata={},
    )


def test_active_contract_persists_across_reload(tmp_path) -> None:
    path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(path))
    contract = _sample_contract()

    manager.set_active_contract("NIFTY", contract)
    saved = manager.get_active_contract("NIFTY")
    assert saved is not None
    assert saved.symbol == contract.symbol

    manager.save_state()

    reloaded = PositionManager(state_file=str(path))
    cached = reloaded.get_active_contract("NIFTY")
    assert cached is not None
    assert cached.symbol == contract.symbol


def test_active_contract_cleared_on_close(tmp_path) -> None:
    path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(path))
    contract = _sample_contract()

    manager.set_active_contract("NIFTY", contract)
    manager.open_position(
        symbol=contract.symbol,
        side="LONG",
        quantity=1,
        entry_price=contract.ltp,
    )

    manager.close_position(contract.symbol, exit_price=130.0, reason="test")

    assert manager.get_active_contract("NIFTY") is None
    assert manager.is_flat(contract.symbol)


class _StubOrderManager:
    def __init__(self) -> None:
        self.guards: list[dict[str, Any]] = []
        self.cleared: list[str] = []

    def reconcile_open_orders_with_broker(self) -> None:
        return None

    def has_guard_pair(self, symbol: str) -> bool:
        return any(entry["symbol"] == symbol for entry in self.guards)

    def guard_existing_position(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        average_price: float,
        last_price: float | None,
        product: str | None,
    ) -> None:
        self.guards.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "average_price": average_price,
                "last_price": last_price,
                "product": product,
            }
        )

    def clear_guard_pair(self, symbol: str) -> None:
        self.cleared.append(symbol)
        self.guards = [entry for entry in self.guards if entry["symbol"] != symbol]


def test_reconcile_updates_active_contract_and_guard(tmp_path) -> None:
    path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(path))
    initial = _sample_contract()
    manager.set_active_contract("NIFTY", initial)

    class _Broker:
        def get_positions(self) -> list[dict[str, Any]]:
            return [
                {
                    "tradingsymbol": "NIFTY25O2025450CE",
                    "product": "MIS",
                    "quantity": 1,
                    "average_price": 125.0,
                    "last_price": 126.5,
                    "strike": 25450.0,
                    "expiry": datetime(2025, 10, 30, tzinfo=timezone.utc),
                }
            ]

    order_manager = _StubOrderManager()
    ctx = SimpleNamespace(
        broker_client=_Broker(),
        position_manager=manager,
        order_manager=order_manager,
    )

    _reconcile_state(ctx)

    cached = manager.get_active_contract("NIFTY")
    assert cached is not None
    assert cached.symbol == "NIFTY25O2025450CE"
    assert order_manager.guards
    assert order_manager.guards[0]["symbol"] == "NIFTY25O2025450CE"


def test_reconcile_rebuilds_guard_after_broker_sync(tmp_path) -> None:
    path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(path))
    broker_positions = [
        {
            "tradingsymbol": "NIFTY25O2025450CE",
            "product": "MIS",
            "quantity": 1,
            "average_price": 125.0,
            "last_price": 126.5,
            "strike": 25450.0,
            "expiry": datetime(2025, 10, 30, tzinfo=timezone.utc),
        }
    ]
    manager.synchronize_with_broker(broker_positions)

    class _Broker:
        def get_positions(self) -> list[dict[str, Any]]:
            return list(broker_positions)

    class _DataHub:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        def replace_positions(self, rows: list[dict[str, Any]]) -> None:
            self.rows = list(rows)

    order_manager = _StubOrderManager()
    data_hub = _DataHub()
    ctx = SimpleNamespace(
        broker_client=_Broker(),
        position_manager=manager,
        order_manager=order_manager,
        data_hub=data_hub,
    )

    _reconcile_state(ctx)

    assert order_manager.guards
    assert order_manager.guards[0]["symbol"] == "NIFTY25O2025450CE"
    cached = manager.get_active_contract("NIFTY")
    assert cached is not None
    assert cached.symbol == "NIFTY25O2025450CE"
    assert data_hub.rows == [
        {
            "symbol": "NIFTY25O2025450CE",
            "quantity": 1,
            "average_price": 125.0,
        }
    ]
