# src/risk/session.py
"""
Manages the state of a single trading session, including P&L,
active trades, and risk limit enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import RiskSettings, settings

logger = logging.getLogger(__name__)


# ----------------------------- Trade model ----------------------------- #


@dataclass
class Trade:
    """Represents a single trade from entry to exit."""

    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    quantity: int  # contracts (NOT lots)
    order_id: str
    atr_at_entry: float
    entry_time: datetime = field(default_factory=datetime.now)

    # Filled on close
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    status: str = "OPEN"  # "OPEN" | "CLOSED"

    def close(self, exit_price: float, fees: float = 0.0) -> None:
        """Mark the trade as closed and calculate P&L."""
        if self.status == "CLOSED":
            logger.warning("Trade %s is already closed.", self.order_id)
            return

        self.exit_price = float(exit_price)
        self.exit_time = datetime.now()

        if self.direction.upper() == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity

        self.fees = float(fees or 0.0)
        self.net_pnl = self.pnl - self.fees
        self.status = "CLOSED"
        logger.info(
            "Closed trade %s (%s). Net P&L: %.2f",
            self.order_id,
            self.symbol,
            self.net_pnl,
        )


# --------------------------- Trading session --------------------------- #


class TradingSession:
    """
    Manages all state for the current trading day, including equity, P&L,
    positions, and risk limits.
    """

    def __init__(
        self,
        risk_config: RiskSettings,
        starting_equity: float,
        fee_per_lot: float = 20.0,  # simple, configurable fee model
        lot_size: int | None = None,
    ):
        if not isinstance(risk_config, RiskSettings):
            raise TypeError("A valid RiskSettings instance is required.")

        self.risk_config = risk_config

        self.start_equity = float(starting_equity)
        self.current_equity = float(starting_equity)

        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trades_today: int = 0

        self.fee_per_lot = float(fee_per_lot)
        self.lot_size = int(lot_size or settings.instruments.nifty_lot_size)

        # order_id -> Trade
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []

    # ---------------------------- trade flow ---------------------------- #

    def add_trade(self, trade: Trade) -> None:
        """Adds a new active trade to the session."""
        if trade.order_id in self.active_trades:
            logger.warning(
                "Attempted to add duplicate trade with order_id: %s", trade.order_id
            )
            return

        self.active_trades[trade.order_id] = trade
        self.trades_today += 1
        logger.info(
            "New trade: %s %d %s @ %.2f (oid=%s)",
            trade.direction,
            trade.quantity,
            trade.symbol,
            trade.entry_price,
            trade.order_id,
        )

    def finalize_trade(self, order_id: str, exit_price: float) -> Trade | None:
        """
        Moves a trade from active to history and updates session P&L.
        Returns the closed Trade or None if not found.
        """
        trade = self.active_trades.pop(order_id, None)
        if not trade:
            logger.warning(
                "Could not find active trade with order_id: %s to finalize.", order_id
            )
            return None

        # fees: simple per-lot model
        fees = self._estimate_fees(trade.quantity)
        trade.close(exit_price=float(exit_price), fees=fees)
        self.trade_history.append(trade)

        # Update session state using net_pnl
        self.daily_pnl += trade.net_pnl
        self.current_equity += trade.net_pnl

        if trade.net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # reset on a winner

        return trade

    # ---------------------------- risk checks --------------------------- #

    @property
    def drawdown_pct(self) -> float:
        """Current intraday drawdown as a fraction of starting equity (>=0)."""
        if self.start_equity <= 0:
            return 0.0
        loss = -min(self.daily_pnl, 0.0)
        return loss / self.start_equity

    def check_risk_limits(self) -> str | None:
        """
        Checks if any session-level risk limits have been breached.
        Returns a string reason if a limit is breached, otherwise None.
        """
        if self.trades_today >= int(self.risk_config.max_trades_per_day):
            return (
                f"Max trades per day ({self.risk_config.max_trades_per_day}) reached."
            )

        if self.consecutive_losses >= int(self.risk_config.consecutive_loss_limit):
            return f"Consecutive loss limit ({self.risk_config.consecutive_loss_limit}) reached."

        if self.drawdown_pct >= float(self.risk_config.max_daily_drawdown_pct):
            return (
                f"Max daily drawdown ({self.risk_config.max_daily_drawdown_pct:.2%}) breached. "
                f"Current DD: {self.drawdown_pct:.2%}"
            )

        return None

    # ------------------------------ helpers ---------------------------- #

    def lots_for_trade(self, equity: float, premium_per_lot: float) -> int:
        """Return lots to trade given equity and per-lot premium.

        Applies the standard exposure cap. When ``RISK__ALLOW_MIN_ONE_LOT`` is
        true and equity can cover one lot outright, returns ``1`` even if the
        exposure cap is smaller than the premium per lot. Otherwise returns
        ``0`` when the cap is insufficient. The override is disabled by default
        and must be explicitly enabled in configuration.
        """
        cap = float(equity) * float(self.risk_config.max_position_size_pct)
        cost_one_lot = float(premium_per_lot)
        allow_min = bool(getattr(self.risk_config, "allow_min_one_lot", False))
        if cap < cost_one_lot:
            if allow_min and equity >= cost_one_lot:
                logger.info(
                    "allow_min_one_lot enabled: cap %.2f < cost %.2f; sizing 1 lot",
                    cap,
                    cost_one_lot,
                )
                return 1
            logger.info(
                "exposure cap %.2f below cost %.2f; sizing 0 lots",
                cap,
                cost_one_lot,
            )
            return 0
        logger.info(
            "exposure cap %.2f allows cost %.2f; sizing 1 lot",
            cap,
            cost_one_lot,
        )
        return 1

    def _estimate_fees(self, quantity_contracts: int) -> float:
        """
        Simple per-lot fee model (flat fee_per_lot per lot).
        Quantity is in contracts; lots = qty / lot_size.
        """
        try:
            lot_size = int(self.lot_size or 0)
            if lot_size <= 0:
                return 0.0
            lots = max(0, quantity_contracts // lot_size)
            return lots * self.fee_per_lot
        except Exception:
            return 0.0

    # Convenience status for UIs/Telegram if needed
    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "session_date": datetime.now().date().isoformat(),
            "account_size": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.active_trades),
            "closed_today": len(self.trade_history),
        }
