"""
Adaptive position sizing module.

This module provides a ``PositionSizing`` class that determines how many
contract lots to trade based on the live account balance from Zerodha,
risk settings, and market conditions. It protects the account by enforcing
per-trade risk limits, a daily drawdown cap, and a maximum number of
consecutive losses.

The calculation assumes that each point move in the underlying contract
is worth ``Config.NIFTY_LOT_SIZE`` rupees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import Config
from src.auth.zerodha_auth import get_kite_client  # fetch live capital

logger = logging.getLogger(__name__)


def get_live_account_balance() -> float:
    try:
        kite = get_kite_client()
        margins = kite.margins(segment='equity')
        cash = margins['available']['cash']
        return float(cash)
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch live account balance, using fallback: {e}")
        return Config.ACCOUNT_SIZE


@dataclass
class PositionSizing:
    """Risk manager that calculates position sizes and tracks drawdown."""

    account_size: float = field(default_factory=get_live_account_balance)
    risk_per_trade: float = Config.RISK_PER_TRADE
    daily_risk: float = Config.MAX_DRAWDOWN
    max_drawdown: float = Config.MAX_DRAWDOWN
    lot_size: int = Config.NIFTY_LOT_SIZE
    min_lots: int = Config.MIN_LOTS
    max_lots: int = Config.MAX_LOTS
    consecutive_loss_limit: int = Config.CONSECUTIVE_LOSS_LIMIT

    # Internal state
    daily_loss: float = 0.0
    equity_peak: float = field(init=False)
    equity: float = field(init=False)
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        self.account_size = get_live_account_balance()
        self.equity = self.account_size
        self.equity_peak = self.account_size
        logger.info(f"💰 Live account size loaded: ₹{self.account_size:.2f}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        market_volatility: float = 0.0,
    ) -> Optional[Dict[str, int]]:
        try:
            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning("❌ Consecutive loss limit reached. No new trades allowed.")
                return None

            sl_points = abs(entry_price - stop_loss)
            if sl_points <= 0:
                logger.warning("⚠️ Stop loss and entry price are equal or invalid.")
                return None

            risk_per_lot = sl_points * self.lot_size
            trade_risk_budget = self.account_size * self.risk_per_trade
            qty = int(trade_risk_budget // risk_per_lot)

            if qty <= 0:
                logger.info(f"❌ Risk per lot ₹{risk_per_lot:.2f} exceeds trade budget ₹{trade_risk_budget:.2f}.")
                return None

            confidence_factor = max(0.1, min(signal_confidence / 10.0, 1.0))
            qty = max(self.min_lots, int(qty * confidence_factor))

            if market_volatility > 0.5:
                qty = max(self.min_lots, qty // 2)

            qty = max(self.min_lots, min(qty, self.max_lots))

            potential_loss = qty * risk_per_lot
            daily_risk_limit = self.account_size * self.daily_risk
            if self.daily_loss + potential_loss > daily_risk_limit:
                logger.warning(
                    f"❌ Daily risk limit exceeded. Trade risk ₹{potential_loss:.2f} + accumulated ₹{self.daily_loss:.2f} > ₹{daily_risk_limit:.2f}"
                )
                return None

            return {"quantity": qty}

        except Exception as exc:
            logger.error(f"💥 Error calculating position size: {exc}")
            return None

    def update_after_trade(self, realised_pnl: float) -> None:
        self.equity += realised_pnl
        self.equity_peak = max(self.equity, self.equity_peak)

        if realised_pnl < 0:
            self.daily_loss += abs(realised_pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        current_drawdown = (self.equity_peak - self.equity) / self.equity_peak
        if current_drawdown >= self.max_drawdown:
            logger.warning(f"❗ Max drawdown reached: {current_drawdown*100:.2f}%. Trading should halt.")

    def reset_daily_limits(self) -> None:
        logger.info("🔄 Resetting daily loss and consecutive loss counters.")
        self.daily_loss = 0.0
        self.consecutive_losses = 0

    def get_risk_status(self) -> Dict[str, float]:
        return {
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "daily_loss": self.daily_loss,
            "consecutive_losses": self.consecutive_losses,
        }

    def update_position_status(self, is_open: bool) -> None:
        logger.debug(f"🧾 Position status updated: is_open={is_open}")
