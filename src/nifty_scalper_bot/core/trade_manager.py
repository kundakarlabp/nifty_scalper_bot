"""Trade state machine and lifecycle manager."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


class TradeState(Enum):
    """Trade lifecycle states."""

    INIT = "INIT"
    ORDER_PLACED = "ORDER_PLACED"
    FILLED = "FILLED"
    ACTIVE = "ACTIVE"
    EXITED = "EXITED"


@dataclass(slots=True)
class Trade:
    """Tracked trade metadata. Args: fields below. Returns: Trade. Raises: ValueError."""

    id: str
    symbol: str
    side: str
    qty: int
    entry_price: float
    sl: float
    target: float
    status: TradeState = TradeState.INIT


class TradeManager:
    """Manage deterministic trade state transitions."""

    def __init__(self) -> None:
        """Initialise the trade manager. Args: none. Returns: None. Raises: None."""
        self._trades_by_id: dict[str, Trade] = {}
        self._open_by_symbol: dict[str, str] = {}
        self._lock = RLock()
        self._valid_transitions: dict[TradeState, set[TradeState]] = {
            TradeState.INIT: {TradeState.ORDER_PLACED},
            TradeState.ORDER_PLACED: {TradeState.FILLED},
            TradeState.FILLED: {TradeState.ACTIVE},
            TradeState.ACTIVE: {TradeState.EXITED},
            TradeState.EXITED: set(),
        }

    def valid_transition(self, old_state: TradeState, new_state: TradeState) -> bool:
        """Return whether transition from old_state to new_state is legal."""
        return new_state in self._valid_transitions.get(old_state, set())

    def has_open_trade(self, symbol: str) -> bool:
        """Check symbol open state. Args: symbol. Returns: bool. Raises: None."""
        with self._lock:
            return symbol.upper() in self._open_by_symbol

    def create_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        sl: float,
        target: float,
    ) -> Trade:
        """Create a new trade. Args: trade values. Returns: Trade. Raises: ValueError."""
        symbol_key = symbol.upper()
        with self._lock:
            if symbol_key in self._open_by_symbol:
                msg = f"Duplicate open trade blocked for {symbol_key}"
                logger.error(
                    msg, extra={"event": "TRADE_REJECTED", "symbol": symbol_key}
                )
                raise ValueError(msg)
            trade = Trade(
                id=trade_id,
                symbol=symbol_key,
                side=side.upper(),
                qty=int(qty),
                entry_price=float(entry_price),
                sl=float(sl),
                target=float(target),
                status=TradeState.INIT,
            )
            self._trades_by_id[trade_id] = trade
            self._open_by_symbol[symbol_key] = trade_id
            self.update_trade(trade_id, TradeState.ORDER_PLACED)
            logger.info(
                '{"event":"ORDER_PLACED","symbol":"%s","trade_id":"%s","qty":%s}',
                symbol_key,
                trade_id,
                qty,
            )
            return trade

    def update_trade(self, trade_id: str, status: TradeState, **updates: Any) -> Trade:
        """Update trade fields. Args: trade_id/status/updates. Returns: Trade. Raises: KeyError."""
        with self._lock:
            trade = self._trades_by_id[trade_id]
            old_state = trade.status
            assert self.valid_transition(old_state, status), (
                f"Invalid trade transition for {trade_id}: "
                f"{old_state.value} -> {status.value}"
            )
            for key, value in updates.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)
            trade.status = status
            logger.info(
                '{"event":"TRADE_STATUS","trade_id":"%s","status":"%s"}',
                trade_id,
                status.value,
            )
            return trade

    def close_trade(
        self, trade_id: str, final_state: TradeState = TradeState.EXITED
    ) -> Trade:
        """Close trade and release symbol lock. Args: trade_id/final_state. Returns: Trade. Raises: KeyError."""
        with self._lock:
            trade = self._trades_by_id[trade_id]
            old_state = trade.status
            assert self.valid_transition(old_state, final_state), (
                f"Invalid trade transition for {trade_id}: "
                f"{old_state.value} -> {final_state.value}"
            )
            trade.status = final_state
            self._open_by_symbol.pop(trade.symbol.upper(), None)
            logger.info(
                '{"event":"POSITION_CLOSED","trade_id":"%s","symbol":"%s","status":"%s"}',
                trade_id,
                trade.symbol,
                final_state.value,
            )
            return trade
