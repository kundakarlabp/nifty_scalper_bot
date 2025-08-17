# src/risk/position_sizing.py
"""
Adaptive position sizing.

- Computes lots from risk budget, SL distance and lot value (contracts/lot).
- Enforces per-trade risk, daily risk (drawdown) and consecutive-loss guard.
- Supports live capital via Zerodha (cached, thread-safe) with a safe fallback.
- Conservative defaults; backward compatible with existing callers.

Public API:
    get_live_account_balance(fallback: float = 30000.0) -> float
    refresh_live_account_balance(force: bool = False) -> float
    class PositionSizing:
        calculate_position_size(...)
        update_after_trade(...)
        reset_daily_limits()
        get_risk_status()
        update_position_status(...)
        refresh_equity_from_live()
        # QoL setters / helpers:
        set_equity(value: float) -> None
        set_risk_per_trade(value: float) -> None
        set_limits(min_lots: int = ..., max_lots: int = ...) -> None
        can_trade_now() -> tuple[bool, str]
"""

from __future__ import annotations

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import Config

# Zerodha client (optional import to keep module robust in shadow/backtests)
try:
    from src.auth.zerodha_auth import get_kite_client  # live capital
except Exception:  # pragma: no cover
    get_kite_client = None  # type: ignore

logger = logging.getLogger(__name__)

# --------------------------- live balance helpers --------------------------- #

_BALANCE_CACHE_VALUE: Optional[float] = None
_BALANCE_CACHE_TS: float = 0.0
_BAL_LOCK = threading.RLock()

_BALANCE_TTL_SEC: int = int(getattr(Config, "BALANCE_TTL_SEC", 60))
_BALANCE_SEGMENT: str = str(getattr(Config, "BALANCE_SEGMENT", "equity")).lower()
_BALANCE_SOURCE: str = str(getattr(Config, "BALANCE_SOURCE", "NET")).upper()  # NET | CASH
_FALLBACK_ACCOUNT_SIZE: float = float(getattr(Config, "ACCOUNT_SIZE", 30000.0))


def _parse_margin_available(avail: dict) -> Optional[float]:
    """
    Prefer 'net' if BALANCE_SOURCE=NET else 'cash'.
    Fall back gracefully across common Zerodha margin payload keys.
    """
    try:
        if not isinstance(avail, dict):
            return None
        if _BALANCE_SOURCE == "NET":
            for k in ("net", "live_balance", "opening_balance"):
                if k in avail and avail[k] is not None:
                    return float(avail[k])
        # CASH path and general fallbacks
        for k in ("cash", "net", "opening_balance", "adhoc_margin"):
            if k in avail and avail[k] is not None:
                return float(avail[k])
    except Exception:
        return None
    return None


def _fetch_live_cash_balance() -> Optional[float]:
    """Low-level fetch from Kite; return None on failure."""
    if get_kite_client is None:  # no live integration available
        return None
    try:
        kite = get_kite_client()
        # Zerodha supports 'equity' or 'commodity'
        margins = kite.margins(segment=_BALANCE_SEGMENT) or {}
        avail = margins.get("available", {}) or {}
        val = _parse_margin_available(avail)
        return float(val) if val is not None else None
    except Exception as e:
        logger.warning("⚠️ Failed to fetch live account balance: %s", e)
        return None


def refresh_live_account_balance(force: bool = False, fallback: Optional[float] = None) -> float:
    """
    Refresh the cached live balance if TTL expired or force=True.
    Returns the cached (possibly refreshed) value.
    """
    global _BALANCE_CACHE_VALUE, _BALANCE_CACHE_TS
    now = time.time()
    with _BAL_LOCK:
        if force or (_BALANCE_CACHE_VALUE is None) or ((now - _BALANCE_CACHE_TS) > _BALANCE_TTL_SEC):
            val = _fetch_live_cash_balance()
            if val is None:
                # Keep previous cache if present; otherwise seed with fallback
                if _BALANCE_CACHE_VALUE is None:
                    _BALANCE_CACHE_VALUE = float(fallback if fallback is not None else _FALLBACK_ACCOUNT_SIZE)
                    logger.info("💰 Using fallback account balance: ₹%.2f", _BALANCE_CACHE_VALUE)
                else:
                    logger.debug("Live balance fetch failed; keeping cached ₹%.2f", _BALANCE_CACHE_VALUE)
            else:
                _BALANCE_CACHE_VALUE = float(val)
                logger.info("💰 Live account balance fetched: ₹%.2f", _BALANCE_CACHE_VALUE)
            _BALANCE_CACHE_TS = now
        return float(_BALANCE_CACHE_VALUE if _BALANCE_CACHE_VALUE is not None else (fallback or _FALLBACK_ACCOUNT_SIZE))


def get_live_account_balance(fallback: float = 30000.0) -> float:
    """
    Get cached live balance; refresh only if TTL expired.
    Returns `fallback` if live fetch fails on cold start.
    Thread-safe.
    """
    return refresh_live_account_balance(force=False, fallback=fallback)


# --------------------------------- class ---------------------------------- #

@dataclass
class PositionSizing:
    """
    Risk manager that calculates position sizes and tracks drawdown.
    """

    account_size: float = field(default_factory=lambda: get_live_account_balance(_FALLBACK_ACCOUNT_SIZE))
    risk_per_trade: float = float(getattr(Config, "RISK_PER_TRADE", 0.01))          # 1% default
    daily_risk: float = float(getattr(Config, "MAX_DRAWDOWN", 0.05))                # day cap proxy
    max_drawdown: float = float(getattr(Config, "MAX_DRAWDOWN", 0.05))              # equity peak drawdown
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
        # Force a fresh balance read on construction (within TTL rules)
        self.account_size = refresh_live_account_balance(fallback=_FALLBACK_ACCOUNT_SIZE)
        self.equity = self.account_size
        self.equity_peak = self.account_size
        # Clamp obvious misconfigs
        self.risk_per_trade = float(max(1e-6, min(self.risk_per_trade, 0.2)))   # <=20% guardrail
        self.daily_risk = float(max(1e-6, min(self.daily_risk, 0.5)))           # <=50% guardrail
        self.max_drawdown = float(max(1e-6, min(self.max_drawdown, 0.9)))       # <=90% guardrail
        self.min_lots = max(0, int(self.min_lots))
        self.max_lots = max(self.min_lots, int(self.max_lots))
        logger.info(
            "💰 Account size set: ₹%.2f (risk/trade=%.2f%%, DD cap=%.2f%%)",
            self.account_size, self.risk_per_trade * 100.0, self.max_drawdown * 100.0
        )

    # --------------------------- core calculation --------------------------- #

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        market_volatility: float = 0.0,
        lot_size: Optional[int] = None,  # override for instrument-specific lots
    ) -> Optional[Dict[str, int]]:
        """
        Calculate number of lots to trade.

        Returns:
            {"quantity": <int>} or None if blocked by risk rules.
        """
        try:
            # Basic validation
            if not (entry_price and stop_loss) or entry_price <= 0 or stop_loss <= 0:
                logger.warning("⚠️ Invalid entry/SL inputs (entry=%.3f, sl=%.3f).", entry_price, stop_loss)
                return None

            # SL distance
            sl_points = abs(float(entry_price) - float(stop_loss))
            if not math.isfinite(sl_points) or sl_points <= 0:
                logger.warning("⚠️ SL distance is zero/invalid.")
                return None

            # Consecutive loss guard
            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning("❌ Consecutive loss limit reached (%d). Blocking new trades.",
                               self.consecutive_loss_limit)
                return None

            # Lot size
            eff_lot_size = int(lot_size) if (lot_size and lot_size > 0) else int(self.lot_size)
            eff_lot_size = max(1, eff_lot_size)

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
                logger.info("❌ Risk/lot ₹%.2f exceeds budget ₹%.2f.", risk_per_lot, trade_risk_budget)
                return None

            qty = int(qty_raw)  # base whole lots

            # --- Adjustments ---

            # 1) Confidence (0–10) → 10%..100% scaling
            conf = max(0.0, min(10.0, float(signal_confidence)))
            confidence_factor = max(0.1, conf / 10.0)
            qty = int(qty * confidence_factor)

            # 2) Volatility curb
            try:
                vol = float(market_volatility)
            except Exception:
                vol = 0.0
            if vol >= 0.75:
                qty = max(self.min_lots, qty // 3)  # very high vol → 1/3rd
            elif vol > 0.5:
                qty = max(self.min_lots, qty // 2)

            # 3) Enforce min/max
            qty = max(self.min_lots, min(qty, self.max_lots))

            if qty <= 0:
                logger.info("❌ Final quantity is zero/negative after adjustments.")
                return None

            # Daily risk cap (pre-trade)
            potential_loss = qty * risk_per_lot
            daily_cap = float(self.account_size) * float(self.daily_risk)
            if (self.daily_loss + potential_loss) > daily_cap:
                logger.warning(
                    "❌ Daily risk limit exceeded. "
                    "Trade risk ₹%.2f + accrued ₹%.2f > cap ₹%.2f",
                    potential_loss, self.daily_loss, daily_cap
                )
                return None

            logger.debug(
                "✅ PosSize=%d lots (SL pts: %.2f, ₹/lot: %.2f, budget: ₹%.2f, conf: %.1f, vol: %.2f)",
                qty, sl_points, risk_per_lot, trade_risk_budget, conf, vol
            )
            return {"quantity": int(qty)}

        except Exception as exc:
            logger.error("💥 Error calculating position size: %s", exc, exc_info=True)
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
            logger.info("📉 Loss: ₹%.2f | Consecutive losses: %d", pnl, self.consecutive_losses)
        else:
            if self.consecutive_losses:
                logger.info("📈 Profit: ₹%.2f | Loss streak reset.", pnl)
            self.consecutive_losses = 0

        dd = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        logger.debug("📊 Equity ₹%.2f | Peak ₹%.2f | DD %.2f%% | Daily Loss ₹%.2f",
                     self.equity, self.equity_peak, dd * 100, self.daily_loss)

        if dd >= float(self.max_drawdown):
            logger.critical("❗ Max drawdown %.2f%% breached (%.2f%%). Halt trading.",
                            self.max_drawdown * 100, dd * 100)
            return False

        return True

    def reset_daily_limits(self) -> None:
        """Reset daily counters; call at start of new trading day."""
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        logger.info("🔄 Daily risk counters reset.")

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

    def update_position_status(self, is_open: bool) -> None:
        # placeholder for compatibility; extend if you later track open positions
        logger.debug("🧾 Position status updated: is_open=%s", bool(is_open))

    # ---------------------------- QoL helpers -------------------------------- #

    def refresh_equity_from_live(self, *, force: bool = False) -> float:
        """
        Refresh internal equity numbers from live balance cache.
        Useful for /refresh or scheduled equity syncs.
        """
        bal = refresh_live_account_balance(force=force, fallback=_FALLBACK_ACCOUNT_SIZE)
        self.set_equity(bal)
        return bal

    def set_equity(self, value: float) -> None:
        """Safely set current equity & adjust peak if needed."""
        try:
            v = max(0.0, float(value))
        except Exception:
            return
        self.equity = v
        self.equity_peak = max(self.equity_peak, v)

    def set_risk_per_trade(self, value: float) -> None:
        """Update risk per trade with guardrails (0..20%)."""
        try:
            v = float(value)
        except Exception:
            return
        self.risk_per_trade = float(max(1e-6, min(v, 0.2)))

    def set_limits(self, *, min_lots: Optional[int] = None, max_lots: Optional[int] = None) -> None:
        """Update min/max lots with sane clamping."""
        if min_lots is not None:
            self.min_lots = max(0, int(min_lots))
        if max_lots is not None:
            self.max_lots = max(self.min_lots, int(max_lots))

    def can_trade_now(self) -> tuple[bool, str]:
        """
        Quick preflight used by a trader loop to short-circuit entries.
        """
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return False, "loss-streak-block"
        daily_cap = float(self.account_size) * float(self.daily_risk)
        if self.daily_loss >= daily_cap:
            return False, "daily-risk-cap"
        dd = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0.0
        if dd >= self.max_drawdown:
            return False, "max-drawdown"
        return True, "ok"