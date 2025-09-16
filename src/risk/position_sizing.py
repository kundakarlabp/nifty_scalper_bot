# Path: src/risk/position_sizing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, cast
import math
import os
import logging
from types import SimpleNamespace
from src.config import settings

logger = logging.getLogger(__name__)


def _mid_from_quote(q: dict) -> float:
    mid = q.get("mid")
    if mid is None:
        b, a = q.get("bid"), q.get("ask")
        if b is not None and a is not None:
            mid = (float(b) + float(a)) / 2.0
        else:
            mid = float(q.get("ltp") or 0.0)
    return float(mid or 0.0)


def lots_from_premium_cap(
    runner,
    quote: dict,
    lot_size: int,
    max_lots: int,
) -> Tuple[int, float, float]:
    from src.config import settings as _settings

    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    cap_abs_raw = _coerce_float(getattr(_settings, "EXPOSURE_CAP_ABS", 0.0))
    cap_abs_setting = float(cap_abs_raw) if cap_abs_raw is not None else 0.0
    risk_cfg = getattr(_settings, "risk", None)
    fallback_equity_raw = _coerce_float(getattr(risk_cfg, "default_equity", 0.0))
    fallback_equity = (
        float(fallback_equity_raw) if fallback_equity_raw is not None else 0.0
    )
    min_equity_floor_raw = _coerce_float(getattr(risk_cfg, "min_equity_floor", 0.0))
    min_equity_floor = (
        float(min_equity_floor_raw) if min_equity_floor_raw is not None else 0.0
    )
    use_live_equity = bool(getattr(risk_cfg, "use_live_equity", True))

    eq_value: float = fallback_equity
    eq_source = "default"

    if _settings.EXPOSURE_CAP_SOURCE == "equity":
        eq_candidate: float | None = None
        if runner is not None and use_live_equity:
            try:
                if hasattr(runner, "get_equity_amount"):
                    eq_candidate = _coerce_float(runner.get_equity_amount())
                elif hasattr(runner, "equity_amount"):
                    eq_candidate = _coerce_float(getattr(runner, "equity_amount"))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("lots_from_premium_cap: live equity fetch failed: %s", exc)
        if eq_candidate is not None:
            eq_value = max(eq_candidate, 0.0)
            eq_source = "live"
        else:
            eq_value = fallback_equity
            eq_source = "default"
            if eq_value <= 0 and min_equity_floor > 0:
                eq_value = min_equity_floor
                eq_source = "floor"
            if fallback_equity > 0:
                eq_value = max(eq_value, fallback_equity)
            if min_equity_floor > 0:
                eq_value = max(eq_value, min_equity_floor)
        eq_value = max(eq_value, 0.0)

        cap_pct = float(_settings.EXPOSURE_CAP_PCT_OF_EQUITY)
        cap_from_pct = max(0.0, eq_value * cap_pct)
        cap = cap_from_pct
        if cap_abs_setting > 0:
            cap = min(cap_from_pct, cap_abs_setting)
        if str(getattr(_settings, "EXPOSURE_BASIS", "premium")).lower() == "premium":
            logger.info(
                "lots_from_premium_cap: basis=premium source=%s eq=%.2f cap_pct=%.2f cap=%.2f cap_abs=%.2f",
                eq_source,
                eq_value,
                cap_pct,
                cap,
                cap_abs_setting,
            )
    else:
        cap = float(_settings.PREMIUM_CAP_PER_TRADE)
        if cap_abs_setting > 0:
            cap = min(cap, cap_abs_setting)

    price = float(_mid_from_quote(quote or {}))
    if price <= 0:
        return 0, 0.0, float(cap)

    unit_notional = price * float(lot_size)
    lots = 0 if unit_notional <= 0 else int(float(cap) // unit_notional)
    lots = min(int(max_lots), max(0, lots))
    return lots, unit_notional, float(cap)


def estimate_r_rupees(entry: float, sl: float, lot_size: int, lots: int) -> float:
    """Return estimated ₹ risk for the given trade parameters."""

    return abs(float(entry) - float(sl)) * int(lot_size) * int(lots)


@dataclass(frozen=True)
class SizingInputs:
    entry_price: float
    stop_loss: float
    lot_size: int
    equity: float
    spot_price: float = 0.0
    spot_sl_points: float = 0.0
    delta: float | None = None


@dataclass(frozen=True)
class SizingParams:
    """
    risk_per_trade: fraction of equity to risk per trade (e.g., 0.01 = 1%, 0.006 = 0.6%)
    min_lots / max_lots: hard clamps on lots
    max_position_size_pct: cap notional exposure vs equity (0.10 = 10%)
    """

    risk_per_trade: float = 0.01
    min_lots: int = 1
    max_lots: int = 10
    max_position_size_pct: float = 0.10
    exposure_basis: Literal["underlying", "premium"] = field(
        default_factory=lambda: cast(
            Literal["underlying", "premium"], settings.EXPOSURE_BASIS
        )
    )
    allow_min_one_lot: bool = field(
        default_factory=lambda: str(
            os.getenv("RISK__ALLOW_MIN_ONE_LOT", "false")
        ).lower()
        in ("1", "true", "yes"),
    )


class PositionSizer:
    """
    Long options position sizing:
      - risk_per_trade is a decimal fraction of equity (e.g., 0.006 = 0.6%)
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
        risk_per_trade: float | None = None,
        min_lots: int | None = None,
        max_lots: int | None = None,
        max_position_size_pct: float | None = None,
        exposure_basis: str | None = None,
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
            min_lots = getattr(rs, "min_lots", None)
            max_lots = getattr(rs, "max_lots", None)
            max_position_size_pct = getattr(rs, "max_position_size_pct", None)
            exposure_basis = getattr(rs, "exposure_basis", None)

        risk_per_trade = float(
            risk_per_trade
            if risk_per_trade is not None
            else os.getenv("RISK__RISK_PER_TRADE_PCT", 0.01)
        )
        min_lots = int(min_lots if min_lots is not None else os.getenv("RISK__MIN_LOTS", 1))
        max_lots = int(max_lots if max_lots is not None else os.getenv("RISK__MAX_LOTS", 10))
        max_position_size_pct = float(
            max_position_size_pct if max_position_size_pct is not None else 0.10
        )
        exposure_basis = exposure_basis or settings.EXPOSURE_BASIS

        if risk_per_trade <= 0 or risk_per_trade > 0.50:
            raise ValueError("risk_per_trade must be within (0, 0.50].")
        if min_lots <= 0 or max_lots <= 0 or max_lots < min_lots:
            raise ValueError(
                "min_lots and max_lots must be positive and max_lots >= min_lots.",
            )
        if max_position_size_pct < 0 or max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be within [0, 1].")

        self.params = SizingParams(
            risk_per_trade=risk_per_trade,
            min_lots=min_lots,
            max_lots=max_lots,
            max_position_size_pct=max_position_size_pct,
            exposure_basis=cast(Literal["underlying", "premium"], str(exposure_basis)),
        )
    @classmethod
    def from_settings(
        cls,
        *,
        risk_per_trade: float,
        min_lots: int,
        max_lots: int,
        max_position_size_pct: float,
        exposure_basis: str = "premium",
    ) -> "PositionSizer":
        """Convenience creator to mirror env/config fields; keeps imports clean here."""
        return cls(
            risk_per_trade=float(risk_per_trade),
            min_lots=int(min_lots),
            max_lots=int(max_lots),
            max_position_size_pct=float(max_position_size_pct),
            exposure_basis=str(exposure_basis),
        )

    def size_from_signal(
        self,
        *,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        equity: float,
        spot_price: float | None = None,
        spot_sl_points: float | None = None,
        delta: float | None = None,
        quote: Dict | None = None,
    ) -> Tuple[int, int, Dict]:
        mid = _mid_from_quote(quote) if quote else float(entry_price)
        sl_points_spot = (
            float(spot_sl_points)
            if spot_sl_points is not None
            else abs(float(spot_price or 0.0) - float(stop_loss))
        )
        si = SizingInputs(
            entry_price=float(mid),
            stop_loss=float(stop_loss),
            lot_size=int(lot_size),
            equity=float(equity),
            spot_price=float(spot_price or 0.0),
            spot_sl_points=sl_points_spot,
            delta=delta,
        )
        qty, lots, diag = self._compute_quantity(si, self.params)
        return qty, lots, diag


    @staticmethod
    def _compute_quantity(si: SizingInputs, sp: SizingParams) -> Tuple[int, int, Dict]:
        if si.entry_price <= 0 or si.lot_size <= 0 or si.equity <= 0:
            return 0, 0, PositionSizer._diag(
                si, sp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "invalid"
            )

        spot_sl_points = abs(si.spot_sl_points)
        if spot_sl_points <= 0:
            return 0, 0, PositionSizer._diag(
                si, sp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "no_sl"
            )

        delta = si.delta if si.delta is not None else 0.5
        delta = max(0.25, min(0.75, delta))
        opt_sl_points = abs(spot_sl_points * delta)
        risk_per_lot_rupees = max(0.5, opt_sl_points) * si.lot_size

        risk_rupees = si.equity * sp.risk_per_trade
        max_lots_risk = (
            int(risk_rupees // risk_per_lot_rupees)
            if risk_per_lot_rupees > 0
            else 0
        )

        cap_abs_setting = float(getattr(settings, "EXPOSURE_CAP_ABS", 0.0) or 0.0)

        if sp.exposure_basis == "premium":
            runner = SimpleNamespace(equity_amount=si.equity)
            max_lots_exposure, unit_notional, exposure_cap = lots_from_premium_cap(
                runner,
                {"mid": si.entry_price},
                si.lot_size,
                sp.max_lots,
            )
            cap_pct = float(settings.EXPOSURE_CAP_PCT_OF_EQUITY)
            min_eq_needed = (
                unit_notional / cap_pct if cap_pct > 0 else float("inf")
            )
            if cap_abs_setting > 0 and unit_notional > cap_abs_setting:
                min_eq_needed = float("inf")
            cap_abs = cap_abs_setting
        else:
            unit_notional = (si.spot_price or si.entry_price) * si.lot_size
            if sp.max_position_size_pct > 0 and unit_notional > 0:
                exposure_cap = si.equity * sp.max_position_size_pct
                max_lots_exposure = int(exposure_cap // unit_notional)
            else:
                exposure_cap = float("inf")
                max_lots_exposure = int(1e12) if unit_notional > 0 else 0
            min_eq_needed = (
                unit_notional / sp.max_position_size_pct
                if sp.max_position_size_pct > 0
                else unit_notional
            )
            cap_abs = 0.0

        max_lots_limit = sp.max_lots
        calc_lots = min(max_lots_exposure, max_lots_risk, max_lots_limit)
        logger.info(
            "sizer calc: exposure=%d risk=%d limit=%d -> calc=%d",
            max_lots_exposure,
            max_lots_risk,
            max_lots_limit,
            calc_lots,
        )
        lots = calc_lots
        if sp.exposure_basis == "premium" and max_lots_exposure < 1:
            if sp.allow_min_one_lot and si.equity >= unit_notional:
                lots = sp.min_lots
                block_reason = ""
                logger.info(
                    "sizer allow_min_one_lot: basis=%s unit=%.2f cap=%.2f -> lots=%d",
                    sp.exposure_basis,
                    unit_notional,
                    exposure_cap,
                    lots,
                )
            else:
                lots = 0
                block_reason = "cap_lt_one_lot"
                logger.info(
                    "sizer block: basis=%s unit=%.2f lots=%d cap=%.2f",
                    sp.exposure_basis,
                    unit_notional,
                    max_lots_exposure,
                    exposure_cap,
                )
        else:
            if (
                lots == 0
                and unit_notional <= exposure_cap
                and sp.min_lots <= max_lots_limit
            ):
                lots = sp.min_lots
                logger.info("sizer clamp to min lots=%d", lots)
            block_reason = ""
            if lots == 0:
                if max_lots_limit < sp.min_lots:
                    block_reason = "lot_limit"
                elif unit_notional > exposure_cap:
                    block_reason = "exposure_cap"
                else:
                    block_reason = "risk_cap"
                logger.info(
                    "sizer block: basis=%s unit=%.2f lots=%d cap=%.2f",
                    sp.exposure_basis,
                    unit_notional,
                    calc_lots,
                    exposure_cap,
                )

        quantity = lots * si.lot_size
        diag = PositionSizer._diag(
            si,
            sp,
            spot_sl_points,
            risk_per_lot_rupees,
            risk_rupees,
            max_lots_exposure,
            max_lots_risk,
            lots,
            unit_notional,
            exposure_cap,
            calc_lots,
            min_eq_needed,
            block_reason,
            cap_abs=cap_abs,
        )
        return quantity, lots, diag

    @staticmethod
    def _diag(
        si: SizingInputs,
        sp: SizingParams,
        spot_sl_points: float,
        risk_per_lot: float,
        risk_rupees: float,
        max_lots_exposure: int,
        max_lots_risk: int,
        lots_final: int,
        unit_notional: float,
        exposure_cap: float,
        calc_lots: int,
        min_equity_needed: float,
        block_reason: str,
        *,
        cap_abs: float = 0.0,
    ) -> Dict:
        unit_val = (
            round(unit_notional, 2)
            if math.isfinite(unit_notional)
            else float("inf")
        )
        cap_val = (
            round(exposure_cap, 2)
            if math.isfinite(exposure_cap)
            else float("inf")
        )
        min_eq_val = (
            round(min_equity_needed, 2)
            if math.isfinite(min_equity_needed)
            else float("inf")
        )
        cap_abs_val = None
        if cap_abs > 0:
            cap_abs_val = (
                round(cap_abs, 2) if math.isfinite(cap_abs) else float("inf")
            )
        return {
            "entry_price": round(si.entry_price, 4),
            "equity": round(si.equity, 2),
            "lot_size": int(si.lot_size),
            "risk_per_trade": float(sp.risk_per_trade),
            "min_lots": int(sp.min_lots),
            "max_lots": int(sp.max_lots),
            "max_position_size_pct": float(sp.max_position_size_pct),
            "spot_sl_points": round(spot_sl_points, 4),
            "delta": None if si.delta is None else float(si.delta),
            "risk_per_lot": round(risk_per_lot, 2),
            "risk_rupees": round(risk_rupees, 2),
            "unit_notional": unit_val,
            "basis": sp.exposure_basis,
            "max_lots_exposure": int(max_lots_exposure),
            "max_lots_risk": int(max_lots_risk),
            "lots_final": int(lots_final),
            "lots": int(lots_final),
            "exposure_cap": cap_val,
            "cap": cap_val,
            "calc_lots": int(calc_lots),
            "min_equity_needed": min_eq_val,
            "block_reason": block_reason,
            "cap_abs": cap_abs_val,
        }
