# Path: src/risk/position_sizing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, cast
import math
import os
import logging
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


def _coerce_float(value: Any) -> float | None:
    """Safely coerce ``value`` to ``float`` when possible."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_default_equity(settings_obj: Any) -> Tuple[float, str]:
    """Resolve fallback equity and identify its source."""

    best_val = 0.0
    best_src = "default"
    for attr in ("RISK_DEFAULT_EQUITY", "risk_default_equity"):
        coerced = _coerce_float(getattr(settings_obj, attr, None))
        if coerced is not None and coerced > best_val:
            best_val = float(coerced)
            best_src = "default"
    risk_cfg = getattr(settings_obj, "risk", None)
    if risk_cfg is not None:
        default_val = _coerce_float(getattr(risk_cfg, "default_equity", None))
        if default_val is not None and default_val > best_val:
            best_val = float(default_val)
            best_src = "default"
        floor_val = _coerce_float(getattr(risk_cfg, "min_equity_floor", None))
        if floor_val is not None and floor_val > best_val:
            best_val = float(floor_val)
            best_src = "floor"
    if best_val > 0:
        return best_val, best_src
    return 0.0, "default"


def _resolve_cap_pct(settings_obj: Any) -> float:
    """Return the exposure cap percentage expressed as 0..100."""

    candidates = (
        getattr(settings_obj, "EXPOSURE_CAP_PCT", None),
        getattr(settings_obj, "EXPOSURE_CAP_PCT_OF_EQUITY", None),
        getattr(settings_obj, "risk_exposure_cap_pct_of_equity", None),
        getattr(getattr(settings_obj, "risk", None), "exposure_cap_pct_of_equity", None),
    )
    for candidate in candidates:
        pct = _coerce_float(candidate)
        if pct is None:
            continue
        if pct <= 1.0:
            pct *= 100.0
        return max(0.0, float(pct))
    return 0.0


def _resolve_cap_abs(settings_obj: Any) -> float:
    """Return absolute premium cap configured on ``settings`` if any."""

    cap_abs = _coerce_float(getattr(settings_obj, "EXPOSURE_CAP_ABS", None))
    if cap_abs is not None:
        return float(cap_abs)
    cap_abs = _coerce_float(getattr(settings_obj, "risk_exposure_cap_abs", None))
    if cap_abs is not None:
        return float(cap_abs)
    risk_cfg = getattr(settings_obj, "risk", None)
    cap_abs = _coerce_float(getattr(risk_cfg, "exposure_cap_abs", None))
    if cap_abs is not None:
        return float(cap_abs)
    return 0.0


def lots_from_premium_cap(
    premium: float,
    lot_size: int,
    settings_obj: Any,
    live_equity: float | None,
) -> Tuple[int, Dict[str, Any]]:
    equity_source = "live"
    equity_val = _coerce_float(live_equity) if live_equity is not None else None
    if equity_val is not None:
        equity = max(float(equity_val), 0.0)
    else:
        equity, equity_source = _resolve_default_equity(settings_obj)

    src_raw = getattr(settings_obj, "EXPOSURE_CAP_SOURCE", "equity") or "equity"
    src = str(src_raw).lower()

    pct = _resolve_cap_pct(settings_obj)
    abs_cap = _resolve_cap_abs(settings_obj)
    if src == "equity":
        cap = equity * pct / 100.0
        if abs_cap > 0:
            cap = min(cap, abs_cap)
    else:
        cap = abs_cap

    unit_notional = float(premium) * float(lot_size)
    lots = int(cap // unit_notional) if unit_notional > 0 else 0

    meta: Dict[str, Any] = {
        "basis": "premium",
        "source": src,
        "equity": float(equity),
        "equity_source": equity_source,
        "cap_pct": float(pct),
        "cap": round(float(cap), 2),
        "cap_abs": round(float(abs_cap), 2) if abs_cap > 0 else None,
        "price": float(premium),
        "lot_size": int(lot_size),
        "unit_notional": round(float(unit_notional), 2),
        "lots": int(lots),
    }

    if unit_notional <= 0:
        meta["reason"] = "unit_notional_le_zero"
        return 0, meta
    if cap < unit_notional:
        meta["reason"] = "cap_lt_one_lot"
        return 0, meta

    return max(lots, 1), meta


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
        cap_meta: Dict[str, Any] | None = None

        if sp.exposure_basis == "premium":
            max_lots_exposure, cap_meta = lots_from_premium_cap(
                premium=si.entry_price,
                lot_size=si.lot_size,
                settings_obj=settings,
                live_equity=si.equity,
            )
            unit_notional = float(
                cap_meta.get("unit_notional", si.entry_price * si.lot_size)
            )
            exposure_cap = float(cap_meta.get("cap", 0.0)) if cap_meta else 0.0
            cap_pct = float(cap_meta.get("cap_pct", 0.0)) if cap_meta else 0.0
            cap_abs_meta = _coerce_float(cap_meta.get("cap_abs")) if cap_meta else None
            if cap_abs_meta is not None and cap_abs_meta > 0:
                cap_abs = float(cap_abs_meta)
            else:
                cap_abs = cap_abs_setting if cap_abs_setting > 0 else 0.0
            if (
                cap_meta is not None
                and cap_meta.get("source") == "equity"
                and cap_pct > 0
            ):
                min_eq_needed = unit_notional * 100.0 / cap_pct
                if cap_abs > 0 and cap_abs < unit_notional:
                    min_eq_needed = float("inf")
            else:
                min_eq_needed = (
                    float("inf") if exposure_cap < unit_notional else unit_notional
                )
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
            cap_meta = None

        max_lots_limit = sp.max_lots
        calc_lots = min(max_lots_exposure, max_lots_risk, max_lots_limit)
        if cap_meta:
            logger.info(
                "sizer calc: exposure=%d risk=%d limit=%d -> calc=%d meta=%s",
                max_lots_exposure,
                max_lots_risk,
                max_lots_limit,
                calc_lots,
                cap_meta,
            )
        else:
            logger.info(
                "sizer calc: exposure=%d risk=%d limit=%d -> calc=%d",
                max_lots_exposure,
                max_lots_risk,
                max_lots_limit,
                calc_lots,
            )
        lots = calc_lots
        if sp.exposure_basis == "premium" and max_lots_exposure < 1:
            reason = "cap_lt_one_lot"
            if cap_meta is not None:
                reason = str(cap_meta.get("reason", reason))
            if (
                sp.allow_min_one_lot
                and si.equity >= unit_notional
                and reason != "unit_notional_le_zero"
            ):
                lots = sp.min_lots
                block_reason = ""
                logger.info(
                    "sizer allow_min_one_lot: basis=%s unit=%.2f cap=%.2f -> lots=%d meta=%s",
                    sp.exposure_basis,
                    unit_notional,
                    exposure_cap,
                    lots,
                    cap_meta,
                )
            else:
                lots = 0
                block_reason = reason
                logger.info(
                    "sizer block: basis=%s unit=%.2f lots=%d cap=%.2f meta=%s",
                    sp.exposure_basis,
                    unit_notional,
                    max_lots_exposure,
                    exposure_cap,
                    cap_meta,
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
                    "sizer block: basis=%s unit=%.2f lots=%d cap=%.2f meta=%s",
                    sp.exposure_basis,
                    unit_notional,
                    calc_lots,
                    exposure_cap,
                    cap_meta,
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
            cap_meta=cap_meta,
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
        cap_meta: Dict[str, Any] | None = None,
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
        cap_meta = cap_meta or {}
        eq_source = cap_meta.get("equity_source") or cap_meta.get("source")
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
            "eq_source": eq_source,
            "cap_meta": cap_meta,
        }
