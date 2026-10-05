"""Underlying entry reservation and re-entry cooldown guard."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _SymbolState:
    direction: str
    last_ts: float


@dataclass(slots=True, frozen=True)
class TradeDecision:
    """Compatibility DTO for the legacy risk-gate boundary."""

    action: str
    direction: str
    symbol: str | None
    underlying: str
    confidence: float
    entry_price: float | None
    stop_loss: float | None
    target: float | None
    rr: float | None
    reasons: list[str]
    votes: list[Any] = field(default_factory=list)
    candidate_meta: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    timestamp: float = 0.0


class SignalArbitrator:
    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        stale_active_seconds: float = 120.0,
        reentry_cooldown_seconds: float = 300.0,
    ) -> None:
        self._cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self._stale_active_seconds = max(
            float(stale_active_seconds), self._cooldown_seconds
        )
        self._reentry_cooldown_seconds = max(
            float(reentry_cooldown_seconds), self._cooldown_seconds
        )
        self._active_symbols: set[str] = set()
        self._state: dict[str, _SymbolState] = {}
        self._lock = threading.RLock()

    def allow(self, signal: Any, action: str | None = None) -> bool:
        if isinstance(signal, str):
            symbol = str(signal or "").upper()
            normalized_action = str(action or "").upper()
        else:
            symbol = str(getattr(signal, "symbol", "") or "").upper()
            normalized_action = str(
                getattr(signal, "action", action or "") or ""
            ).upper()
        key = self._reservation_key(symbol)
        if not key:
            return False
        direction = self._direction(normalized_action)
        now = time.time()
        with self._lock:
            prev = self._state.get(key)
            if key in self._active_symbols:
                if prev is not None and now - prev.last_ts >= self._stale_active_seconds:
                    self._active_symbols.discard(key)
                else:
                    return False
            if prev is None:
                return True
            elapsed = now - prev.last_ts
            if direction and prev.direction == direction:
                return elapsed >= self._reentry_cooldown_seconds
            return elapsed >= self._cooldown_seconds

    def register(self, symbol: str, action: str = "") -> None:
        key = self._reservation_key(symbol)
        if not key:
            return
        with self._lock:
            self._active_symbols.add(key)
            self._state[key] = _SymbolState(
                direction=self._direction(action), last_ts=time.time()
            )

    def release(self, symbol: str) -> None:
        key = self._reservation_key(symbol)
        if not key:
            return
        with self._lock:
            self._active_symbols.discard(key)
            state = self._state.get(key)
            if state is not None:
                state.last_ts = time.time()

    def clear(self, symbol: str) -> None:
        """Drop a reservation for an entry that never created exposure."""
        key = self._reservation_key(symbol)
        if not key:
            return
        with self._lock:
            self._active_symbols.discard(key)
            self._state.pop(key, None)

    @staticmethod
    def _reservation_key(symbol: str) -> str:
        """Collapse NIFTY option contracts to one underlying entry reservation."""
        normalized = str(symbol or "").upper().split(":")[-1]
        for underlying in ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTY"):
            if normalized.startswith(underlying):
                return underlying
        return re.sub(r"\s+", "", normalized)

    @staticmethod
    def _direction(action: str) -> str:
        if action in {"BUY", "CLOSE_SHORT"}:
            return "LONG"
        if action in {"SELL", "CLOSE_LONG"}:
            return "SHORT"
        return ""
