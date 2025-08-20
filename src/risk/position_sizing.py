# src/risk/position_sizing.py
"""
Adaptive position sizing.

- Computes lots from risk budget, SL distance and lot value.
- Enforces per-trade risk, daily risk (drawdown) and consecutive-loss guard.
- Supports live capital via Zerodha (cached) with a safe fallback.

This file is intentionally conservative about changes to preserve
compatibility with existing callers (e.g., RealTimeTrader).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import settings
from src.auth.zerodha_auth import get_kite_client  # fetch live capital

logger = logging.getLogger(__name__)

# --------------------------- live balance helpers --------------------------- #

_BALANCE_CACHE_VALUE: Optional[float] = None
_BALANCE_CACHE_TS: float = 0.0
_BALANCE_TTL_SEC: int = 60  # refresh at most once per minute


def _fetch_live_cash_balance() -> Optional[float]:
    """Low-level fetch from Kite; return None on failure."""
    try:
        kite = get_kite_client()
        margins = kite.margins(segment="equity")
        cash = margins["available"]["cash"]
        return float(cash)
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch live account balance: {e}")
        return None


def get_live_account_balance(fallback: float = 30000.0) -> float:
    """
    Get cached live balance; refresh only if TTL expired.
    Returns `fallback` if live fetch fails.
    """
    global _BALANCE_CACHE_VALUE, _BALANCE_CACHE_TS
    now = time.time()
    if _BALANCE_CACHE_VALUE is None or (now - _BALANCE_CACHE_TS) > _BALANCE_TTL_SEC:
        val = _fetch_live_cash_balance()
        if val is None:
            if _BALANCE_CACHE_VALUE is None:
                logger.info(f"💰 Using fallback account balance: ₹{fallback:.2f}")
                _BALANCE_CACHE_VALUE = float(fallback)
                _BALANCE_CACHE_TS = now
        else:
            _BALANCE_CACHE_VALUE = float(val)
            _BALANCE_CACHE_TS = now
            logger.info(f"💰 Live account balance fetched: ₹{_BALANCE_CACHE_VALUE:.2f}")
    return float(_BALANCE_CACHE_VALUE or fallback)


# --------------------------------- class ---------------------------------- #

@dataclass
class PositionSizing:
    """
    Risk manager that calculates position sizes and tracks drawdown.
    """

    account_size: float = field(default_factory=lambda: get_live_account_balance())
    risk_per_trade: float = float(getattr(Config, "RISK_PER_TRADE", 0.01))
    daily_risk: float = float(getattr(Config, "MAX_DRAWDOWN", 0.05))
    max_drawdown: float = float(getattr(Config, "MAX_DRAWDOWN", 0.05))
    lot_size: int = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    min_lots: int = int(getattr(Config, "MIN_LOTS", 1))
    max_lots: int = int(getattr(Config, "MAX_LOTS", 5))
    consecutive_loss_limit: int = int(getattr(Config, "CONSECUTIVE_LOSS_LIMIT", 3))

    # Internal state
    daily_loss: float = 0.0
    equity_peak: float = field(init=False)
    equity: float = field(init=False)
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        self.account_size = float(max(0.0, self.account_size))
        self.equity = self.account_size
        self.equity_peak = self.account_size
        logger.info(f"💰 Account size set: ₹{self.account_size:.2f}")

    # --------------------------- core calculation --------------------------- #

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        market_volatility: float = 0.0,
        lot_size: Optional[int] = None,  # <-- optional override for options path
    ) -> Optional[Dict[str, int]]:
        """
        Calculate number of lots to trade.

        Args:
            entry_price: Proposed entry price.
            stop_loss: Absolute stop price (not distance).
            signal_confidence: 0–10 scale (used to scale qty).
            market_volatility: 0.0–1.0 (high volatility shrinks qty).
            lot_size: Optional override for the contract lot size.

        Returns:
            {"quantity": <int>} or None if blocked by risk rules.
        """
        try:
            # Basic validation
            if not (entry_price and stop_loss) or entry_price <= 0 or stop_loss <= 0:
                logger.warning("⚠️ Invalid entry/SL inputs.")
                return None

            # Prevent divide-by-zero / nonsense
            sl_points = abs(float(entry_price) - float(stop_loss))
            if not math.isfinite(sl_points) or sl_points <= 0:
                logger.warning("⚠️ SL distance is zero/invalid.")
                return None

            # Consecutive loss guard
            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning("❌ Consecutive loss limit reached. Blocking new trades.")
                return None

            # Lot size to use
            eff_lot_size = int(lot_size) if (lot_size and lot_size > 0) else int(self.lot_size)

            # Risk per lot (₹ per point * points to SL)
            risk_per_lot = sl_points * eff_lot_size
            if risk_per_lot <= 0 or not math.isfinite(risk_per_lot):
                logger.warning("⚠️ Computed risk_per_lot invalid.")
                return None

            # Trade risk budget
            trade_risk_budget = float(self.account_size) * float(self.risk_per_trade)
            if trade_risk_budget <= 0:
                logger.warning("⚠️ Risk budget is zero/negative; check config/account size.")
                return None

            # Raw qty from budget
            qty_raw = trade_risk_budget / risk_per_lot
            if qty_raw <= 0:
                logger.info(
                    f"❌ Risk per lot ₹{risk_per_lot:.2f} exceeds budget ₹{trade_risk_budget:.2f}."
                )
                return None

            qty = int(qty_raw)  # base whole lots

            # --- Adjustments ---

            # 1) Confidence (0–10) → 10%..100% scaling
            conf = max(0.0, min(10.0, float(signal_confidence)))
            confidence_factor = max(0.1, conf / 10.0)
            qty = max(0, int(qty * confidence_factor))

            # 2) Volatility curb
            try:
                vol = float(market_volatility)
            except Exception:
                vol = 0.0
            if vol > 0.5:
                qty = max(self.min_lots, qty // 2)

            # 3) Enforce min/max
            qty = max(self.min_lots, min(qty, self.max_lots))

            if qty <= 0:
                logger.info("❌ Final quantity is zero/negative after adjustments.")
                return None

            # Daily risk cap
            potential_loss = qty * risk_per_lot
            daily_cap = float(self.account_size) * float(self.daily_risk)
            if (self.daily_loss + potential_loss) > daily_cap:
                logger.warning(
                    "❌ Daily risk limit exceeded. "
                    f"Trade risk ₹{potential_loss:.2f} + accrued ₹{self.daily_loss:.2f} > cap ₹{daily_cap:.2f}"
                )
                return None

            logger.debug(
                f"✅ Position size = {qty} lots "
                f"(SL pts: {sl_points:.2f}, risk/lot: ₹{risk_per_lot:.2f}, "
                f"budget: ₹{trade_risk_budget:.2f}, conf: {conf:.1f}, vol: {vol:.2f})"
            )
            return {"quantity": int(qty)}

        except Exception as exc:
            logger.error(f"💥 Error calculating position size: {exc}", exc_info=True)
            return None

    # ---------------------------- state management --------------------------- #

    def update_after_trade(self, realised_pnl: float) -> bool:
        """
        Update risk state after a trade is closed.

        Returns False if trading should halt (e.g., drawdown breached).
        """
        pnl = float(realised_pnl or 0.0)
        self.equity += pnl
        self.equity_peak = max(self.equity_peak, self.equity)

        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
            logger.info(
                f"📉 Loss: ₹{pnl:.2f}. Consecutive losses = {self.consecutive_losses}"
            )
        else:
            self.consecutive_losses = 0
            logger.info(f"📈 Profit: ₹{pnl:.2f}. Loss streak reset.")

        dd = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        logger.debug(
            f"📊 Equity: ₹{self.equity:.2f}, Peak: ₹{self.equity_peak:.2f}, "
            f"Drawdown: {dd*100:.2f}%, Daily Loss: ₹{self.daily_loss:.2f}"
        )

        if dd >= float(self.max_drawdown):
            logger.critical(
                f"❗ Max drawdown {self.max_drawdown*100:.2f}% breached ({dd*100:.2f}%). Halt trading."
            )
            return False

        return True

    def reset_daily_limits(self) -> None:
        """Reset daily counters; call at start of new trading day."""
        logger.info("🔄 Resetting daily risk counters.")
        self.daily_loss = 0.0
        self.consecutive_losses = 0

    def get_risk_status(self) -> Dict[str, float]:
        """Expose current risk metrics (for /status)."""
        dd = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        return {
            "equity": float(self.equity),
            "equity_peak": float(self.equity_peak),
            "current_drawdown": float(dd),
            "daily_loss": float(self.daily_loss),
            "consecutive_losses": float(self.consecutive_losses),
            "risk_level": float(dd),  # simple scalar; UI decides labels
        }

    # placeholder for compatibility; extend if you later track open positions
    def update_position_status(self, is_open: bool) -> None:
        logger.debug(f"🧾 Position status updated: is_open={bool(is_open)}")