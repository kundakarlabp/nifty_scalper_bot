"""Execution router: LIVE / PAPER / SHADOW routing."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .bracket import BracketManager, ExitSignal
from .order_state import Order, OrderSide, OrderStatus, OrderType, Position, PositionSide

if TYPE_CHECKING:
    from ..broker.base import BrokerGateway
    from ..config.settings import TradingSettings
    from ..notifications.notifier import Notifier
    from ..strategies.signal import Signal

log = logging.getLogger(__name__)


class ExecutionRouter:
    """Routes signals to LIVE broker, PAPER broker, or SHADOW log."""

    def __init__(
        self,
        broker: "BrokerGateway",
        settings: "TradingSettings",
        bracket_manager: BracketManager,
        notifier: "Notifier | None" = None,
    ) -> None:
        self._broker = broker
        self._mode = settings.execution_mode.value
        self._brackets = bracket_manager
        self._notifier = notifier
        self._positions: dict[str, Position] = {}
        self._shadow_pnl: float = 0.0

    async def route(self, signal: "Signal") -> Position | None:
        if self._mode == "SHADOW":
            await self._shadow_track(signal)
            return None
        elif self._mode in ("LIVE", "PAPER"):
            return await self._execute(signal)
        return None

    async def _execute(self, signal: "Signal") -> Position | None:
        try:
            option_symbol = f"{signal.symbol}{signal.expiry.strftime('%y%b').upper()}{int(signal.strike)}{signal.option_type}"
            order_id = str(uuid.uuid4())
            pos_id = str(uuid.uuid4())

            # Place order via broker (real or paper)
            result = await self._broker.place_order(
                symbol=option_symbol,
                side="BUY",
                quantity=signal.quantity,
                order_type=signal.metadata.get("order_type", "MARKET"),
            )

            fill_price = result.get("average_price", signal.entry_price)

            # Create position
            pos = Position(
                position_id=pos_id,
                symbol=option_symbol,
                underlying=signal.symbol,
                option_type=signal.option_type,
                strike=signal.strike,
                expiry=signal.expiry,
                side=PositionSide.LONG,
                quantity=signal.quantity,
                initial_quantity=signal.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                pnl_unrealized=0.0,
                pnl_realized=0.0,
                strategy_name=signal.strategy_name,
                regime_at_entry=signal.regime,
                opened_at=datetime.now(timezone.utc),
                last_updated_at=datetime.now(timezone.utc),
            )
            self._positions[pos_id] = pos

            # Create bracket
            self._brackets.create_bracket(pos_id, signal, fill_price)

            if self._notifier:
                await self._notifier.send_fill_alert(pos, signal)

            log.info("Position opened: %s %s @ %.2f", signal.direction.value, option_symbol, fill_price)
            return pos

        except Exception as e:
            log.error("Execution failed for %s: %s", signal.strategy_name, e)
            return None

    async def handle_exit(self, exit_signal: "ExitSignal") -> None:
        """Place exit order for lifecycle-triggered exits."""
        try:
            await self._broker.place_order(
                symbol=exit_signal.symbol,
                side="SELL",
                quantity=exit_signal.quantity,
                order_type="MARKET",
            )
            pos = self._positions.get(exit_signal.position_id)
            if pos and exit_signal.reason in ("SL", "TP2", "TRAIL", "TIME"):
                pos.is_closed = True
            log.info("Exit %s: %s qty=%d", exit_signal.reason, exit_signal.symbol, exit_signal.quantity)
        except Exception as e:
            log.error("Exit failed %s: %s", exit_signal.symbol, e)

    async def _shadow_track(self, signal: "Signal") -> None:
        log.info("SHADOW signal: %s %s conf=%.2f entry=%.2f",
                 signal.direction.value, signal.symbol,
                 signal.confidence, signal.entry_price)

    def get_open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_closed]
