# Path: src/risk/position_sizing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


def estimate_r_rupees(entry: float, sl: float, lot_size: int, lots: int) -> float:
    """Return estimated ₹ risk for the given trade parameters."""

    return abs(float(entry) - float(sl)) * int(lot_size) * int(lots)


@dataclass(frozen=True)
class SizingInputs:
    entry_price: float
    stop_loss: float
    lot_size: int
    equity: float


@dataclass(frozen=True)
class SizingParams:
    """
    risk_per_trade: fraction of equity to risk per trade (0.01 = 1%)
    min_lots / max_lots: hard clamps on lots
    max_position_size_pct: cap notional exposure vs equity (0.10 = 10%)
    """

    risk_per_trade: float = 0.01
    min_lots: int = 1
    max_lots: int = 10
    max_position_size_pct: float = 0.10


class PositionSizer:
    """
    Long options position sizing:
      - risk_rupees = equity * risk_per_trade
      - rupee risk per 1 lot = sl_points * lot_size
      - lots_raw = floor(risk_rupees / rupee_risk_per_lot)
      - if 1 lot exceeds risk_rupees → size 0 (SAFE)
      - else clamp to [min_lots, max_lots]
      - exposure cap: entry_price * lot_size * lots <= equity * max_position_size_pct
      - quantity = lots * lot_size
    Returns (quantity, lots, diagnostics).
    """

    def __init__(
        self,
        risk_per_trade,
        min_lots: int | None = None,
        max_lots: int | None = None,
        max_position_size_pct: float | None = None,
    ) -> None:
        """Create a PositionSizer.

        Accepts either a RiskSettings-like object as the first argument or the
        individual parameters explicitly. This maintains backward compatibility
        with callers that previously passed the risk settings object directly.
        """

        if (
            min_lots is None
            and max_lots is None
            and max_position_size_pct is None
            and hasattr(risk_per_trade, "risk_per_trade")
        ):
            rs = risk_per_trade
            risk_per_trade = getattr(rs, "risk_per_trade")
            min_lots = getattr(rs, "min_lots", 1)
            max_lots = getattr(rs, "max_lots", 10)
            max_position_size_pct = getattr(rs, "max_position_size_pct", 0.10)

        if (
            min_lots is None
            or max_lots is None
            or max_position_size_pct is None
            or risk_per_trade is None
        ):
            raise TypeError(
                "PositionSizer requires either a risk settings object or explicit parameters"
            )

        if risk_per_trade <= 0 or risk_per_trade > 0.10:
            raise ValueError("risk_per_trade must be within (0, 0.10].")
        if min_lots <= 0 or max_lots <= 0 or max_lots < min_lots:
            raise ValueError(
                "min_lots and max_lots must be positive and max_lots >= min_lots."
            )
        if max_position_size_pct < 0 or max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be within [0, 1].")

        self.params = SizingParams(
            risk_per_trade=risk_per_trade,
            min_lots=min_lots,
            max_lots=max_lots,
            max_position_size_pct=max_position_size_pct,
        )

    @classmethod
    def from_settings(
        cls,
        *,
        risk_per_trade: float,
        min_lots: int,
        max_lots: int,
        max_position_size_pct: float,
    ) -> "PositionSizer":
        """Convenience creator to mirror env/config fields; keeps imports clean here."""
        return cls(
            risk_per_trade=float(risk_per_trade),
            min_lots=int(min_lots),
            max_lots=int(max_lots),
            max_position_size_pct=float(max_position_size_pct),
        )

    def size_from_signal(
        self,
        *,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        equity: float,
    ) -> Tuple[int, int, Dict]:
        si = SizingInputs(
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            lot_size=int(lot_size),
            equity=float(equity),
        )
        qty, lots, diag = self._compute_quantity(si, self.params)
        return qty, lots, diag

    @staticmethod
    def _compute_quantity(si: SizingInputs, sp: SizingParams) -> Tuple[int, int, Dict]:
        # Basic input validation
        if (
            si.entry_price <= 0
            or si.stop_loss <= 0
            or si.lot_size <= 0
            or si.equity <= 0
        ):
            return 0, 0, PositionSizer._diag(si, sp, 0, 0, 0, 0, 0, 0, 0)

        sl_points = abs(si.entry_price - si.stop_loss)
        if sl_points <= 0:
            return 0, 0, PositionSizer._diag(si, sp, 0, 0, 0, 0, 0, 0, 0)

        # Risk math
        risk_rupees = si.equity * sp.risk_per_trade
        rupee_risk_per_lot = sl_points * si.lot_size
        if rupee_risk_per_lot <= 0:
            return (
                0,
                0,
                PositionSizer._diag(si, sp, sl_points, 0, risk_rupees, 0, 0, 0, 0),
            )

        # Lots affordable under risk budget
        lots_raw = int(risk_rupees // rupee_risk_per_lot)

        # If the risk budget can't fund even a single lot, exit early
        if lots_raw < 1:
            return (
                0,
                0,
                PositionSizer._diag(
                    si,
                    sp,
                    sl_points,
                    rupee_risk_per_lot,
                    risk_rupees,
                    lots_raw,
                    0,
                    0,
                    0,
                ),
            )

        # Enforce min/max only when at least one lot is affordable
        lots = max(lots_raw, sp.min_lots)
        lots = min(lots, sp.max_lots)

        # Exposure cap (notional)
        exposure_notional = si.entry_price * si.lot_size * lots
        max_notional = (
            si.equity * sp.max_position_size_pct
            if sp.max_position_size_pct > 0
            else float("inf")
        )
        if exposure_notional > max_notional:
            denom = si.entry_price * si.lot_size
            lots_cap = int(max_notional // denom) if denom > 0 else 0
            lots = max(min(lots_cap, lots), 0)
            # recompute after cap so diagnostics reflect the final state
            exposure_notional = si.entry_price * si.lot_size * lots

        quantity = lots * si.lot_size
        diag = PositionSizer._diag(
            si,
            sp,
            sl_points,
            rupee_risk_per_lot,
            risk_rupees,
            lots_raw,
            lots,
            exposure_notional,
            0.0 if max_notional == float("inf") else max_notional,
        )
        return quantity, lots, diag

    @staticmethod
    def _diag(
        si: SizingInputs,
        sp: SizingParams,
        sl_points: float,
        rupee_risk_per_lot: float,
        risk_rupees: float,
        lots_raw: int,
        lots_capped: int,
        exposure_notional: float,
        max_notional: float,
    ) -> Dict:
        # Round floats for compact Telegram display and logs
        return {
            "entry_price": round(si.entry_price, 4),
            "stop_loss": round(si.stop_loss, 4),
            "equity": round(si.equity, 2),
            "lot_size": int(si.lot_size),
            "risk_per_trade": float(sp.risk_per_trade),
            "min_lots": int(sp.min_lots),
            "max_lots": int(sp.max_lots),
            "max_position_size_pct": float(sp.max_position_size_pct),
            "sl_points": round(sl_points, 4),
            "rupee_risk_per_lot": round(rupee_risk_per_lot, 2),
            "risk_rupees": round(risk_rupees, 2),
            "lots_raw": int(lots_raw),
            "lots_final": int(lots_capped),
            "exposure_notional_est": round(exposure_notional, 2),
            "max_notional_cap": round(max_notional, 2),
        }
