"""Signal arbitration guard. Args: strategy signals. Returns: allow/deny. Raises: None."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _SymbolState:
    """Per-symbol arbitration state. Args: direction/last_ts. Returns: state. Raises: None."""

    direction: str
    last_ts: float


class SignalArbitrator:
    """Lightweight signal collision arbitrator. Args: cooldown_seconds. Returns: None. Raises: None."""

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self._cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self._active_symbols: set[str] = set()
        self._state: dict[str, _SymbolState] = {}
        self._lock = threading.RLock()

    def allow(self, signal: Any) -> bool:
        """Check if a signal can pass. Args: signal. Returns: bool. Raises: None."""
        symbol = str(getattr(signal, 'symbol', '') or '').upper()
        action = str(getattr(signal, 'action', '') or '').upper()
        if not symbol:
            return False
        direction = self._direction(action)
        now = time.time()
        with self._lock:
            if symbol in self._active_symbols:
                return False
            prev = self._state.get(symbol)
            if prev is None:
                return True
            if direction and prev.direction == direction:
                return False
            if now - prev.last_ts < self._cooldown_seconds:
                return False
            return True

    def register(self, symbol: str, action: str = '') -> None:
        """Register symbol as active. Args: symbol/action. Returns: None. Raises: None."""
        key = str(symbol or '').upper()
        if not key:
            return
        direction = self._direction(action)
        with self._lock:
            self._active_symbols.add(key)
            self._state[key] = _SymbolState(direction=direction, last_ts=time.time())

    def release(self, symbol: str) -> None:
        """Release active symbol lock. Args: symbol. Returns: None. Raises: None."""
        key = str(symbol or '').upper()
        if not key:
            return
        with self._lock:
            self._active_symbols.discard(key)

    @staticmethod
    def _direction(action: str) -> str:
        """Map action to direction. Args: action. Returns: str. Raises: None."""
        if action in {'BUY', 'CLOSE_SHORT'}:
            return 'LONG'
        if action in {'SELL', 'CLOSE_LONG'}:
            return 'SHORT'
        return ''
