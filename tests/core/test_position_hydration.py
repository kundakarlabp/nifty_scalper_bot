from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.core.app import _hydrate_positions


class _PersistentState:
    def __init__(self, positions: list[Any]) -> None:
        self._positions = list(positions)

    def load_positions(self) -> list[Any]:
        return list(self._positions)


class _PositionManager:
    def __init__(self) -> None:
        self.restored: list[Any] | None = None
        self.synced: list[dict[str, Any]] | None = None
        self._positions: list[Any] = []
        self.broker_client: Any | None = None

    def set_broker_client(self, broker_client: Any | None) -> None:
        self.broker_client = broker_client

    def restore_positions(self, positions: list[Any]) -> None:
        self.restored = list(positions)
        self._positions = list(positions)

    def synchronize_with_broker(self, positions: list[dict[str, Any]]) -> None:
        self.synced = list(positions)
        self._positions = [
            SimpleNamespace(
                symbol=str(entry["tradingsymbol"]).upper(),
                side="LONG" if int(entry["quantity"]) > 0 else "SHORT",
                quantity=abs(int(entry["quantity"])),
                entry_price=float(entry["average_price"]),
            )
            for entry in positions
            if int(entry["quantity"]) != 0
        ]

    def get_open_positions(self) -> list[Any]:
        return list(self._positions)


class _DataHub:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def replace_positions(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)


def test_hydrate_positions_prefers_broker_snapshot(
    monkeypatch,
) -> None:
    persisted = [
        SimpleNamespace(
            symbol="PERSISTED",
            side="LONG",
            quantity=2,
            entry_price=99.0,
        )
    ]
    broker_positions = [
        {
            "tradingsymbol": "NIFTY25O2025450CE",
            "quantity": 1,
            "average_price": 125.0,
        }
    ]
    manager = _PositionManager()
    data_hub = _DataHub()
    monkeypatch.setattr(
        "nifty_scalper_bot.core.app._fetch_positions_with_retry",
        lambda *args, **kwargs: broker_positions,
    )

    broker_client = object()

    result = _hydrate_positions(
        position_manager=manager,
        persistent_state=_PersistentState(persisted),
        broker_client=broker_client,
        data_hub=data_hub,
        max_attempts=1,
        backoff_min=0.1,
        backoff_max=0.1,
        backoff_multiplier=1.0,
        jitter_fraction=0.0,
        total_timeout_sec=1.0,
    )

    assert result == broker_positions
    assert manager.broker_client is broker_client
    assert manager.restored is None
    assert manager.synced == broker_positions
    assert data_hub.rows == [
        {
            "symbol": "NIFTY25O2025450CE",
            "quantity": 1,
            "average_price": 125.0,
        }
    ]


def test_hydrate_positions_falls_back_to_persisted_state(
    monkeypatch,
) -> None:
    persisted = [
        SimpleNamespace(
            symbol="NIFTY25O2025450PE",
            side="SHORT",
            quantity=1,
            entry_price=88.5,
        )
    ]
    manager = _PositionManager()
    data_hub = _DataHub()

    def _raise_fetch(*args, **kwargs) -> list[dict[str, Any]]:
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(
        "nifty_scalper_bot.core.app._fetch_positions_with_retry",
        _raise_fetch,
    )

    result = _hydrate_positions(
        position_manager=manager,
        persistent_state=_PersistentState(persisted),
        broker_client=object(),
        data_hub=data_hub,
        max_attempts=1,
        backoff_min=0.1,
        backoff_max=0.1,
        backoff_multiplier=1.0,
        jitter_fraction=0.0,
        total_timeout_sec=1.0,
    )

    assert result is None
    assert manager.synced is None
    assert manager.restored == persisted
    assert data_hub.rows == [
        {
            "symbol": "NIFTY25O2025450PE",
            "quantity": -1,
            "average_price": 88.5,
        }
    ]
