# src/risk/position_sizing.py
from __future__ import annotations
"""
PositionSizing — production-grade, broker-agnostic sizing for LONG NIFTY OPTIONS

What this module does
- Computes order quantity (in units) for long options based on risk per trade and stop distance.
- Uses caller-supplied equity (so you can pass LIVE equity from the runner), with a safe fallback.
- Enforces min/max lots and an exposure cap (% of equity) using rough notional.
- Exposes a diagnostic helper to print all intermediate numbers for quick Telegram/CLI checks.

Key design choices
- No broker imports. The runner (or caller) supplies `equity` and `lot_size`.
- Stateless, pure functions + a thin class for ergonomics.
- Works with any option instrument where quantity must be a multiple of lot_size.

Typical usage (from runner)
----------------------------------------------------------------
from src.risk.position_sizing import PositionSizer

sizer = PositionSizer(
    risk_per_trade=settings.risk_risk_per_trade,
    min_lots=settings.instruments_min_lots,
    max_lots=settings.instruments_max_lots,
    max_position_size_pct=settings.risk_max_position_size_pct,
)

equity = active_equity_snapshot  # live or fallback
qty, lots, diag = sizer.size_from_signal(
    entry_price=signal.entry_price,
    stop_loss=signal.stop_loss,
    lot_size=settings.instruments_nifty_lot_size,
    equity=equity,
)

if qty > 0:
    # pass `qty` into the executor
----------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SizingInputs:
    entry_price: float         # option entry price (₹)
    stop_loss: float           # option stop loss price (₹)
    lot_size: int              # instrument lot size (e.g., 75 for NIFTY options)
    equity: float              # account equity to base risk on (live or fallback)


@dataclass(frozen=True)
class SizingParams:
    risk_per_trade: float = 0.01           # fraction of equity (e.g., 0.01 = 1%)
    min_lots: int = 1                      # floor on lots
    max_lots: int = 10                     # ceiling on lots
    max_position_size_pct: float = 0.10    # cap notional ≈ entry * lot_size * lots vs equity


class PositionSizer:
    """
    Thin convenience wrapper. Uses pure static methods under the hood.
    """
    def __init__(
        self,
        risk_per_trade: float,
        min_lots: int,
        max_lots: int,
        max_position_size_pct: float,
    ) -> None:
        if risk_per_trade <= 0 or risk_per_trade > 0.10:
            raise ValueError("risk_per_trade must be within (0, 0.10].")
        if min_lots <= 0 or max_lots <= 0 or max_lots < min_lots:
            raise ValueError("min_lots and max_lots must be positive and max_lots >= min_lots.")
        if max_position_size_pct < 0 or max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be within [0, 1].")

        self.params = SizingParams(
            risk_per_trade=risk_per_trade,
            min_lots=min_lots,
            max_lots=max_lots,
            max_position_size_pct=max_position_size_pct,
        )

    # ------------- Public API -------------

    def size_from_signal(
        self,
        *,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        equity: float,
    ) -> Tuple[int, int, Dict]:
        """
        Returns:
            quantity (int): units to send to broker (multiple of lot_size)
            lots (int): computed lots (pre-multiplied)
            diagnostic (dict): intermediate numbers for logging/Telegram
        """
        si = SizingInputs(
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            lot_size=int(lot_size),
            equity=float(equity),
        )
        qty, lots, diag = self._compute_quantity(si, self.params)
        return qty, lots, diag

    # ------------- Core logic -------------

    @staticmethod
    def _compute_quantity(
        si: SizingInputs,
        sp: SizingParams,
    ) -> Tuple[int, int, Dict]:
        """
        Long options sizing:
            risk_rupees = equity * risk_per_trade
            sl_points = |entry - stop|
            rupee_risk_per_lot = sl_points * lot_size
            lots = floor(risk_rupees / rupee_risk_per_lot), clipped to [min_lots, max_lots]
            exposure cap: notional ≈ entry * lot_size * lots <= equity * max_position_size_pct
        """
        # Sanity checks
        if si.entry_price <= 0 or si.stop_loss <= 0 or si.lot_size <= 0 or si.equity <= 0:
            return 0, 0, PositionSizer._diag(si, sp, sl_points=0.0, rupee_risk_per_lot=0.0,
                                             risk_rupees=0.0, lots_raw=0, lots_capped=0,
                                             exposure_notional=0.0, max_notional=0.0)

        sl_points = abs(si.entry_price - si.stop_loss)
        if sl_points <= 0:
            return 0, 0, PositionSizer._diag(si, sp, sl_points=0.0, rupee_risk_per_lot=0.0,
                                             risk_rupees=0.0, lots_raw=0, lots_capped=0,
                                             exposure_notional=0.0, max_notional=0.0)

        risk_rupees = si.equity * sp.risk_per_trade
        rupee_risk_per_lot = sl_points * si.lot_size

        if rupee_risk_per_lot <= 0:
            return 0, 0, PositionSizer._diag(si, sp, sl_points=sl_points, rupee_risk_per_lot=0.0,
                                             risk_rupees=risk_rupees, lots_raw=0, lots_capped=0,
                                             exposure_notional=0.0, max_notional=0.0)

        lots_raw = int(risk_rupees // rupee_risk_per_lot)
        lots = max(lots_raw, sp.min_lots)
        lots = min(lots, sp.max_lots)

        # Exposure cap (approximate)
        exposure_notional = si.entry_price * si.lot_size * lots
        max_notional = si.equity * sp.max_position_size_pct if sp.max_position_size_pct > 0 else float("inf")
        if exposure_notional > max_notional:
            # Recalculate lots under the cap:
            #   lots_cap ≈ floor(max_notional / (entry * lot_size))
            # Ensure non-negative and respect min lots if possible
            denom = si.entry_price * si.lot_size
            lots_cap = int(max_notional // denom) if denom > 0 else 0
            lots = max(min(lots_cap, lots), 0)

        quantity = lots * si.lot_size
        diag = PositionSizer._diag(
            si, sp,
            sl_points=sl_points,
            rupee_risk_per_lot=rupee_risk_per_lot,
            risk_rupees=risk_rupees,
            lots_raw=lots_raw,
            lots_capped=lots,
            exposure_notional=exposure_notional,
            max_notional=max_notional if max_notional != float("inf") else 0.0,
        )
        return quantity, lots, diag

    # ------------- Diagnostics -------------

    @staticmethod
    def _diag(
        si: SizingInputs,
        sp: SizingParams,
        *,
        sl_points: float,
        rupee_risk_per_lot: float,
        risk_rupees: float,
        lots_raw: int,
        lots_capped: int,
        exposure_notional: float,
        max_notional: float,
    ) -> Dict:
        return {
            "entry_price": round(si.entry_price, 4),
            "stop_loss": round(si.stop_loss, 4),
            "equity": round(si.equity, 2),
            "lot_size": si.lot_size,
            "risk_per_trade": sp.risk_per_trade,
            "min_lots": sp.min_lots,
            "max_lots": sp.max_lots,
            "max_position_size_pct": sp.max_position_size_pct,
            "sl_points": round(sl_points, 4),
            "rupee_risk_per_lot": round(rupee_risk_per_lot, 2),
            "risk_rupees": round(risk_rupees, 2),
            "lots_raw": lots_raw,
            "lots_final": lots_capped,
            "exposure_notional_est": round(exposure_notional, 2),
            "max_notional_cap": round(max_notional, 2),
        }
