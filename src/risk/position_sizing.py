# Path: src/risk/position_sizing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, cast
import logging
import math

from src.config import settings
from src.logs import structured_log

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
    """Best-effort conversion of ``value`` to ``float``."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_equity_for_cap(
    *, settings_obj: Any, equity_live: float | None
) -> Tuple[float, str]:
    """Return the equity amount and source used for premium caps."""

    risk_cfg = getattr(settings_obj, "risk", None)
    fallback_equity = float(
        _coerce_float(getattr(risk_cfg, "default_equity", 0.0)) or 0.0
    )
    min_equity_floor = float(
        _coerce_float(getattr(risk_cfg, "min_equity_floor", 0.0)) or 0.0
    )
    use_live_equity = bool(getattr(risk_cfg, "use_live_equity", True))

    equity_value = max(fallback_equity, 0.0)
    equity_source = "default"

    equity_candidate: float | None = None
    if use_live_equity:
        equity_candidate = _coerce_float(equity_live)
    if equity_candidate is not None and math.isfinite(equity_candidate):
        equity_value = max(equity_candidate, 0.0)
        equity_source = "live"
    else:
        candidates = [
            ("default", max(fallback_equity, 0.0)),
            ("floor", max(min_equity_floor, 0.0)),
        ]
        equity_source, equity_value = max(candidates, key=lambda item: item[1])
        if equity_value <= 0.0:
            equity_source = "default"
            equity_value = max(equity_value, 0.0)

    return equity_value, equity_source


def lots_from_premium_cap(
    runner,
    quote: dict,
    lot_size: int,
    max_lots: int,
) -> Tuple[int, float, float, str]:
    from src.config import settings as _settings

    price = float(_mid_from_quote(quote or {}))
    lot_size_int = int(lot_size)

    equity_live: float | None = None
    use_live_equity = bool(getattr(getattr(_settings, "risk", None), "use_live_equity", True))
    if runner is not None and use_live_equity:
        try:
            if hasattr(runner, "get_equity_amount"):
                equity_live = _coerce_float(runner.get_equity_amount())
            elif hasattr(runner, "equity_amount"):
                equity_live = _coerce_float(getattr(runner, "equity_amount"))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("lots_from_premium_cap: live equity fetch failed: %s", exc)

    sizer = PositionSizer()
    cap_info = sizer.compute_exposure_cap(
        equity_live=equity_live,
        unit_premium=price,
        lot_size=lot_size_int,
    )

    exposure_cap = float(cap_info.get("cap", 0.0))
    unit_notional = float(cap_info.get("one_lot_cost", 0.0))
    lots_affordable = int(cap_info.get("lots_affordable", 0))
    lots = min(int(max_lots), max(0, lots_affordable))
    eq_source = str(cap_info.get("eq_source") or cap_info.get("equity_source") or "default")

    if cap_info.get("basis") == "premium":
        logger.info(
            "lots_from_premium_cap: basis=premium source=%s eq=%.2f cap_pct=%.2f cap=%.2f cap_abs=%.2f",
            eq_source,
            float(cap_info.get("equity") or 0.0),
            float(cap_info.get("cap_pct") or 0.0),
            exposure_cap,
            float(cap_info.get("cap_abs") or 0.0),
        )

    if price <= 0:
        unit_notional = 0.0

    return lots, unit_notional, exposure_cap, eq_source


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
    max_position_size_pct: cap notional exposure vs equity (e.g., 0.55 = 55%)
    """

    risk_per_trade: float = 0.01
    min_lots: int = 1
    max_lots: int = 10
    max_position_size_pct: float = 0.55
    exposure_basis: Literal["underlying", "premium"] = field(
        default_factory=lambda: cast(
            Literal["underlying", "premium"], settings.EXPOSURE_BASIS
        )
    )
    allow_min_one_lot: bool = False


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

        risk_cfg = getattr(settings, "risk", None)
        inst_cfg = getattr(settings, "instruments", None)

        allow_min_one_lot = bool(getattr(risk_cfg, "allow_min_one_lot", False))

        if (
            min_lots is None
            and max_lots is None
            and max_position_size_pct is None
            and hasattr(risk_per_trade, "risk_per_trade")
        ):
            rs = risk_per_trade
            allow_min_one_lot = bool(getattr(rs, "allow_min_one_lot", allow_min_one_lot))
            risk_per_trade = getattr(rs, "risk_per_trade")
            min_lots = getattr(rs, "min_lots", None)
            max_lots = getattr(rs, "max_lots", None)
            max_position_size_pct = getattr(rs, "max_position_size_pct", None)
            exposure_basis = getattr(rs, "exposure_basis", None)

        if risk_per_trade is None:
            risk_per_trade = getattr(risk_cfg, "risk_per_trade", 0.01)
        if min_lots is None:
            min_lots = getattr(inst_cfg, "min_lots", 1)
        if max_lots is None:
            max_lots = getattr(inst_cfg, "max_lots", 10)
        if max_position_size_pct is None:
            max_position_size_pct = getattr(risk_cfg, "max_position_size_pct", 0.10)
        if exposure_basis is None:
            exposure_basis = getattr(risk_cfg, "exposure_basis", settings.EXPOSURE_BASIS)

        risk_per_trade = float(risk_per_trade)
        min_lots = int(min_lots)
        max_lots = int(max_lots)
        max_position_size_pct = float(max_position_size_pct)
        exposure_basis = str(exposure_basis)

        if risk_per_trade <= 0 or risk_per_trade > 0.50:
            raise ValueError("risk_per_trade must be within (0, 0.50].")
        if min_lots <= 0 or max_lots <= 0 or max_lots < min_lots:
            raise ValueError(
                "min_lots and max_lots must be positive and max_lots >= min_lots.",
            )
        if max_position_size_pct < 0 or max_position_size_pct > 1:
            raise ValueError("max_position_size_pct must be within [0, 1].")

        self.settings = settings
        self.params = SizingParams(
            risk_per_trade=risk_per_trade,
            min_lots=min_lots,
            max_lots=max_lots,
            max_position_size_pct=max_position_size_pct,
            exposure_basis=cast(Literal["underlying", "premium"], exposure_basis),
            allow_min_one_lot=allow_min_one_lot,
        )

    def compute_exposure_cap(
        self,
        *,
        equity_live: float | None,
        unit_premium: float,
        lot_size: int,
    ) -> Dict[str, Any]:
        """Compute the exposure cap, cost of one lot, and affordable lots."""

        s = self.settings
        basis = str(getattr(s, "EXPOSURE_BASIS", "premium")).lower()
        cap_source = str(getattr(s, "EXPOSURE_CAP_SOURCE", "equity")).lower()

        cap_abs_setting = float(
            _coerce_float(getattr(s, "EXPOSURE_CAP_ABS", 0.0)) or 0.0
        )
        cap_pct_setting = _coerce_float(
            getattr(s, "EXPOSURE_CAP_PCT_OF_EQUITY", None)
        )
        if cap_pct_setting is None:
            cap_pct_setting = _coerce_float(
                getattr(getattr(s, "risk", None), "exposure_cap_pct_of_equity", None)
            )
        cap_pct_setting = float(cap_pct_setting or 0.0)
        if cap_pct_setting > 1.0:
            cap_pct_fraction = cap_pct_setting / 100.0
        else:
            cap_pct_fraction = cap_pct_setting

        lot_size_int = int(lot_size)
        unit_price = float(_coerce_float(unit_premium) or 0.0)
        one_lot_cost = max(0.0, unit_price * lot_size_int)

        equity_value: float | None = None
        equity_source: str | None = None
        cap_value: float
        lots_affordable: int

        if basis != "premium":
            cap_value = float("inf")
            lots_affordable = int(self.params.max_lots)
        else:
            if cap_source == "equity":
                equity_value, equity_source = _resolve_equity_for_cap(
                    settings_obj=s, equity_live=equity_live
                )
                cap_value = max(
                    0.0,
                    (equity_value or 0.0) * cap_pct_fraction,
                )
            else:
                static_cap = _coerce_float(getattr(s, "PREMIUM_CAP_PER_TRADE", None))
                if static_cap is None:
                    static_cap = _coerce_float(
                        getattr(
                            getattr(s, "risk", None), "premium_cap_per_trade", None
                        )
                    )
                cap_value = max(0.0, float(static_cap or 0.0))
                equity_source = cap_source

            if cap_abs_setting > 0.0 and math.isfinite(cap_value):
                cap_value = min(cap_value, cap_abs_setting)

            if math.isfinite(cap_value) and one_lot_cost > 0:
                lots_affordable = int(cap_value // one_lot_cost)
            elif math.isfinite(cap_value):
                lots_affordable = 0
            else:
                lots_affordable = int(self.params.max_lots)

        min_equity_needed = float("inf")
        if basis == "premium":
            if cap_pct_fraction > 0 and one_lot_cost > 0:
                min_equity_needed = one_lot_cost / cap_pct_fraction
            if cap_abs_setting > 0 and one_lot_cost > cap_abs_setting:
                min_equity_needed = float("inf")

        cap_abs_val = cap_abs_setting if cap_abs_setting > 0 else None

        return {
            "basis": basis,
            "cap_source": cap_source,
            "cap": float(cap_value),
            "cap_abs": cap_abs_val,
            "cap_pct": cap_pct_fraction,
            "equity": None if equity_value is None else float(equity_value),
            "eq_source": equity_source,
            "one_lot_cost": float(one_lot_cost),
            "lots_affordable": int(max(0, lots_affordable)),
            "min_equity_for_one_lot": min_equity_needed,
        }
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


    def _compute_quantity(self, si: SizingInputs, sp: SizingParams) -> Tuple[int, int, Dict]:
        if si.entry_price <= 0 or si.lot_size <= 0 or si.equity <= 0:
            diag = PositionSizer._diag(
                si, sp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "invalid"
            )
            try:
                structured_log.event(
                    "sizing_fail",
                    reason="invalid_inputs",
                    diag=diag,
                )
            except Exception:  # pragma: no cover - defensive logging guard
                pass
            return 0, 0, diag

        spot_sl_points = abs(si.spot_sl_points)
        if spot_sl_points <= 0:
            diag = PositionSizer._diag(
                si, sp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "no_sl"
            )
            try:
                structured_log.event(
                    "sizing_fail",
                    reason="no_stop_loss",
                    diag=diag,
                )
            except Exception:  # pragma: no cover - defensive logging guard
                pass
            return 0, 0, diag

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

        if sp.exposure_basis == "premium":
            cap_info = self.compute_exposure_cap(
                equity_live=si.equity,
                unit_premium=si.entry_price,
                lot_size=si.lot_size,
            )
            exposure_cap = float(cap_info.get("cap", 0.0))
            unit_notional = float(cap_info.get("one_lot_cost", 0.0))
            lots_affordable = int(cap_info.get("lots_affordable", 0))
            max_lots_exposure = max(0, min(lots_affordable, sp.max_lots))
            min_eq_needed = float(
                cap_info.get("min_equity_for_one_lot", float("inf"))
            )
            cap_abs = float(cap_info.get("cap_abs") or 0.0)
            eq_source = str(cap_info.get("eq_source") or "na")
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
            eq_source = "na"

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
            eq_source=eq_source,
            cap_abs=cap_abs,
        )
        event_name = "sizing_calc" if lots > 0 else "sizing_fail"
        try:
            structured_log.event(
                event_name,
                lots=int(lots),
                quantity=int(quantity),
                block_reason=diag.get("block_reason"),
                diag=diag,
            )
        except Exception:  # pragma: no cover - defensive logging guard
            pass
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
        eq_source: str | None = None,
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
            "eq_source": eq_source,
        }
