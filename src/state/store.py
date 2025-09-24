"""Simple JSON-backed state store for open orders and positions."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PersistedState:
    """In-memory representation of persisted trading state."""

    open_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


class StateStore:
    """Persist and restore open orders/positions across restarts."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.state = PersistedState()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:  # pragma: no cover - best effort load
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.state = PersistedState(
                open_orders=dict(data.get("open_orders", {})),
                positions=dict(data.get("positions", {})),
            )
        except FileNotFoundError:  # pragma: no cover - startup
            self.state = PersistedState()
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt
            self.state = PersistedState()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._lock, open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f)

    # ------------------------------------------------------------------
    def record_order(self, client_oid: str, payload: dict[str, Any]) -> None:
        """Persist a pending order payload keyed by client OID."""  # pragma: no cover - I/O
        self.state.open_orders[client_oid] = dict(payload)
        self._save()

    def remove_order(self, client_oid: str) -> None:
        if client_oid in self.state.open_orders:  # pragma: no cover - I/O
            del self.state.open_orders[client_oid]
            self._save()

    def has_order(self, client_oid: str) -> bool:
        """Return True if an open order with ``client_oid`` exists."""
        return client_oid in self.state.open_orders

    def record_position(self, symbol: str, info: dict[str, Any]) -> None:
        """Persist an open position snapshot."""  # pragma: no cover - I/O
        self.state.positions[symbol] = dict(info)
        self._save()

    def remove_position(self, symbol: str) -> None:
        if symbol in self.state.positions:  # pragma: no cover - I/O
            del self.state.positions[symbol]
            self._save()

    def snapshot(self) -> PersistedState:
        """Return the current in-memory snapshot."""
        return self.state
