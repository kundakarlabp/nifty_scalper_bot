"""Central risk limits engine handling pre and post trade checks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, cast
from zoneinfo import ZoneInfo

from src.config import settings
from src.risk.position_sizing import _mid_from_quote, lots_from_premium_cap

logger = logging.getLogger(__name__)


@dataclass
class LimitConfig:
    """Configuration knobs for :class:`RiskEngine`."""

    tz: str = "Asia/Kolkata"
    max_daily_dd_R: float = 2.5
    max_trades_per_session: int = 40
    max_lots_per_symbol: int = 5
    max_notional_rupees: float = 1_500_000.0
    exposure_basis: Literal["underlying", "premium"] = field(
        default_factory=lambda: cast(
            Literal["underlying", "premium"], settings.EXPOSURE_BASIS
        )
    )
    max_gamma_mode_lots: int = 2
    max_portfolio_delta_units: int = 100
    max_portfolio_delta_units_gamma: int = 60
    roll10_pause_R: float = -0.2
    roll10_pause_minutes: int = 60
    # Maximum consecutive losing trades before a cooloff is enforced.
    # `cooloff_losses` is kept for backwards compatibility.
    cooloff_losses: int = 3
    max_consec_losses: int = 3
    cooloff_minutes: int = 45
    skip_next_open_after_two_daily_caps: bool = True
    # Maximum realised loss in premium (rupee) terms before halting for the day.
    max_daily_loss_rupees: float = 1_000_000.0
    no_new_after_hhmm: Optional[str] = field(
        default_factory=lambda: settings.risk.no_new_after_hhmm
    )
    eod_flatten_hhmm: str = field(
        default_factory=lambda: str(settings.risk.eod_flatten_hhmm)
    )
    var_lookback_trades: int = 30
    var_confidence: float = 0.95
    cvar_confidence: float = 0.975
    max_var_rupees: Optional[float] = None
    max_var_pct_of_equity: Optional[float] = None
    max_cvar_rupees: Optional[float] = None
    max_cvar_pct_of_equity: Optional[float] = None
    volatility_ref_atr_pct: float = 2.0
    volatility_loss_min_multiplier: float = 0.5
    volatility_loss_max_multiplier: float = 1.3
    regime_lot_multipliers: Dict[str, float] = field(default_factory=dict)
    regime_delta_multipliers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_consec_losses == 3 and self.cooloff_losses != 3:
            self.max_consec_losses = self.cooloff_losses


@dataclass
class Exposure:
    """Snapshot of current market exposure."""

    lots_by_symbol: Dict[str, int] = field(default_factory=dict)
    notional_rupees: float = 0.0


@dataclass(frozen=True)
class Block:
    """Structured risk block result returned by :meth:`RiskEngine.pre_trade_check`."""

    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        """Allow tuple-style unpacking ``ok, reason, details = Block(...)``."""

        yield False
        yield self.reason
        yield self.details

    def __bool__(self) -> bool:
        """A block result is always falsy to halt downstream execution."""

        return False

    def as_tuple(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Explicit tuple representation for legacy call sites and tests."""

        return False, self.reason, self.details


@dataclass
class RiskState:
    """Mutable state tracked by :class:`RiskEngine`."""

    tz: str = "Asia/Kolkata"
    session_date: Optional[str] = None
    cum_R_today: float = 0.0
    cum_loss_rupees: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    roll_R_last10: List[float] = field(default_factory=list)
    cooloff_until: Optional[datetime] = None
    daily_caps_hit_recent: List[str] = field(default_factory=list)
    skip_next_open_date: Optional[str] = None
    pnl_history_rupees: List[float] = field(default_factory=list)

    def new_session_if_needed(self, now_ist: datetime) -> None:
        """Reset per-session counters if the date changed."""

        d = now_ist.date().isoformat()
        if self.session_date != d:
            self.session_date = d
            self.cum_R_today = 0.0
            self.cum_loss_rupees = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.cooloff_until = None


class RiskEngine:
    """Risk gatekeeper performing pre-trade checks and post-trade updates."""

    def __init__(self, cfg: LimitConfig) -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.tz)
        self.state = RiskState(cfg.tz)
        self._max_var_history = max(cfg.var_lookback_trades * 4, 200)

    # -------- helpers --------
    def _now(self) -> datetime:
        return datetime.now(self.tz)

    def _is_gamma_mode(self, now: datetime) -> bool:
        # Tuesday (1) at or after 14:45 IST
        return now.weekday() == 1 and (now.time() >= time(14, 45))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _dynamic_loss_cap(self, plan: dict) -> Optional[float]:
        atr_pct_raw = plan.get("atr_pct_raw") or plan.get("atr_pct")
        if atr_pct_raw is None:
            atr_pct_raw = plan.get("opt_atr_pct")
        if atr_pct_raw is None:
            return None
        try:
            atr_pct_val = float(atr_pct_raw)
        except (TypeError, ValueError):
            return None
        if atr_pct_val <= 0 or self.cfg.volatility_ref_atr_pct <= 0:
            return None
        ref = float(self.cfg.volatility_ref_atr_pct)
        ratio = ref / atr_pct_val
        multiplier = self._clamp(
            ratio,
            float(self.cfg.volatility_loss_min_multiplier),
            float(self.cfg.volatility_loss_max_multiplier),
        )
        return abs(self.cfg.max_daily_loss_rupees) * multiplier

    def _regime_key(self, plan: dict) -> str:
        regime = str(plan.get("regime", "")).strip().upper()
        return regime or ""

    def _lot_cap_for(self, symbol: str, regime: str) -> Tuple[int, str]:
        cap = int(self.cfg.max_lots_per_symbol)
        source = "global"
        try:
            inst = settings.instruments.instrument(symbol)
        except AttributeError:
            inst = None
        except Exception:  # pragma: no cover - defensive guard
            inst = None
        if inst is not None:
            inst_cap = getattr(inst, "max_lots", None)
            if inst_cap:
                cap = min(cap, int(inst_cap))
                source = "instrument" if cap == int(inst_cap) else "global"
        multiplier = None
        if regime:
            multiplier = self.cfg.regime_lot_multipliers.get(regime)
        if multiplier is not None:
            cap = int(math.floor(cap * float(multiplier)))
            source = "regime" if source == "global" else f"{source}+regime"
        return max(cap, 0), source

    def _delta_cap_for(self, regime: str, *, gamma_mode: bool) -> Tuple[int, Optional[str]]:
        cap = (
            self.cfg.max_portfolio_delta_units_gamma
            if gamma_mode
            else self.cfg.max_portfolio_delta_units
        )
        multiplier = None
        if regime:
            multiplier = self.cfg.regime_delta_multipliers.get(regime)
        if multiplier is None:
            return cap, None
        adjusted = int(math.floor(cap * float(multiplier)))
        return max(adjusted, 0), "regime"

    @staticmethod
    def _quantile(data: List[float], q: float) -> float:
        if not data:
            return 0.0
        q = max(0.0, min(1.0, q))
        if q == 0:
            return float(data[0])
        if q == 1:
            return float(data[-1])
        pos = (len(data) - 1) * q
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return float(data[int(pos)])
        lower_val = float(data[lower])
        upper_val = float(data[upper])
        return lower_val + (upper_val - lower_val) * (pos - lower)

    def _var_metrics(self) -> Optional[Dict[str, float]]:
        if self.cfg.var_lookback_trades <= 0:
            return None
        history = self.state.pnl_history_rupees[-self.cfg.var_lookback_trades :]
        if len(history) < self.cfg.var_lookback_trades:
            return None
        losses = sorted(-p for p in history if p < 0)
        if not losses:
            return {"var": 0.0, "cvar": 0.0}
        var_q = self._quantile(losses, float(self.cfg.var_confidence))
        cvar_conf = max(float(self.cfg.cvar_confidence), float(self.cfg.var_confidence))
        cvar_cut = self._quantile(losses, cvar_conf)
        tail = [loss for loss in losses if loss >= cvar_cut]
        if not tail:
            tail = [cvar_cut]
        cvar_val = sum(tail) / len(tail)
        return {"var": float(var_q), "cvar": float(cvar_val)}

    @staticmethod
    def _resolve_threshold(*values: Optional[float]) -> Optional[float]:
        candidates = [abs(v) for v in values if v is not None and v > 0]
        if not candidates:
            return None
        return min(candidates)

    def _var_thresholds(self, equity_rupees: float) -> Tuple[Optional[float], Optional[float]]:
        var_cap_pct = (
            equity_rupees * float(self.cfg.max_var_pct_of_equity)
            if self.cfg.max_var_pct_of_equity is not None and equity_rupees > 0
            else None
        )
        cvar_cap_pct = (
            equity_rupees * float(self.cfg.max_cvar_pct_of_equity)
            if self.cfg.max_cvar_pct_of_equity is not None and equity_rupees > 0
            else None
        )
        var_threshold = self._resolve_threshold(self.cfg.max_var_rupees, var_cap_pct)
        cvar_threshold = self._resolve_threshold(
            self.cfg.max_cvar_rupees, cvar_cap_pct
        )
        return var_threshold, cvar_threshold

    # -------- PRE-TRADE CHECK --------
    def pre_trade_check(
        self,
        *,
        now: datetime | None = None,
        equity_rupees: float,
        plan: dict,
        runner: Optional[Any] = None,
        exposure: Exposure,
        intended_symbol: str,
        intended_lots: int,
        lot_size: int,
        entry_price: float,
        stop_loss_price: float,
        spot_price: float,
        quote: Optional[Dict[str, float]] = None,
        option_mid_price: Optional[float] = None,
        planned_delta_units: Optional[float] = None,
        portfolio_delta_units: Optional[float] = None,
        gamma_mode: Optional[bool] = None,
    ) -> Block | Tuple[bool, str, Dict[str, Any]]:
        """Return (ok, reason_block, details) for a proposed trade."""

        when = now
        if when is None:
            when = self._now()
        elif when.tzinfo is None:
            when = when.replace(tzinfo=self.tz)
        else:
            when = when.astimezone(self.tz)
        now = when
        self.state.new_session_if_needed(now)

        details: Dict[str, Any] = {}
        regime = self._regime_key(plan)
        if regime:
            details["regime"] = regime
        dynamic_loss_cap = self._dynamic_loss_cap(plan)
        if dynamic_loss_cap is not None:
            details["dynamic_loss_cap"] = round(dynamic_loss_cap, 2)

        if self.state.cooloff_until and now < self.state.cooloff_until:
            return (
                False,
                "loss_cooloff",
                {"until": self.state.cooloff_until.isoformat()},
            )

        if self.state.skip_next_open_date == now.date().isoformat():
            return False, "skip_next_open", {}

        if self.state.cum_R_today <= -self.cfg.max_daily_dd_R:
            sd = self.state.session_date or ""
            if sd not in self.state.daily_caps_hit_recent:
                self.state.daily_caps_hit_recent.append(sd)
            if (
                self.cfg.skip_next_open_after_two_daily_caps
                and len(self.state.daily_caps_hit_recent) >= 2
                and self.state.daily_caps_hit_recent[-1] == self.state.session_date
            ):
                self.state.skip_next_open_date = (
                    now.date() + timedelta(days=1)
                ).isoformat()
            return (
                False,
                "daily_dd_hit",
                {"cum_R_today": round(self.state.cum_R_today, 2)},
            )

        loss_threshold = -abs(self.cfg.max_daily_loss_rupees)
        loss_reason = "daily_premium_loss"
        if dynamic_loss_cap is not None:
            loss_threshold = -abs(dynamic_loss_cap)
            loss_reason = "volatility_loss_cap"
        if self.state.cum_loss_rupees <= loss_threshold:
            return (
                False,
                loss_reason,
                {
                    "cum_loss_rupees": round(self.state.cum_loss_rupees, 2),
                    "threshold": round(abs(loss_threshold), 2),
                },
            )

        var_metrics = self._var_metrics()
        if var_metrics is not None:
            var_threshold, cvar_threshold = self._var_thresholds(equity_rupees)
            details["var_estimate"] = round(var_metrics["var"], 2)
            details["cvar_estimate"] = round(var_metrics["cvar"], 2)
            if var_threshold is not None and var_metrics["var"] >= var_threshold:
                return (
                    False,
                    "var_limit",
                    {
                        "estimate": round(var_metrics["var"], 2),
                        "threshold": round(var_threshold, 2),
                    },
                )
            if cvar_threshold is not None and var_metrics["cvar"] >= cvar_threshold:
                return (
                    False,
                    "cvar_limit",
                    {
                        "estimate": round(var_metrics["cvar"], 2),
                        "threshold": round(cvar_threshold, 2),
                    },
                )

        if self.state.trades_today >= self.cfg.max_trades_per_session:
            return (
                False,
                "max_trades_session",
                {"trades_today": self.state.trades_today},
            )

        cutoff_cfg = (self.cfg.no_new_after_hhmm or "").strip()
        if cutoff_cfg and cutoff_cfg.lower() != "none":
            cutoff = datetime.strptime(cutoff_cfg, "%H:%M").time()
            if now.time() >= cutoff:
                return (
                    False,
                    "session_closed",
                    {
                        "cutoff": cutoff_cfg,
                        "config_key": "RISK__NO_NEW_AFTER_HHMM",
                    },
                )

        current_lots = exposure.lots_by_symbol.get(intended_symbol, 0)
        lot_cap, lot_source = self._lot_cap_for(intended_symbol, regime)
        details["lot_cap"] = lot_cap
        details["lot_cap_source"] = lot_source
        if lot_cap <= 0:
            return (
                False,
                "lot_cap",
                {"sym": intended_symbol, "lots": current_lots, "cap": lot_cap},
            )
        if current_lots >= lot_cap:
            reason = "max_lots_symbol" if lot_source == "global" else "lot_cap"
            return (
                False,
                reason,
                {
                    "sym": intended_symbol,
                    "lots": current_lots,
                    "intended": intended_lots,
                    "cap": lot_cap,
                },
            )
        basis_cfg = str(self.cfg.exposure_basis)
        settings_basis = str(settings.EXPOSURE_BASIS)
        use_premium_basis = (
            settings_basis.lower() == "premium" and basis_cfg == "premium"
        )
        basis = "premium" if use_premium_basis else basis_cfg
        exposure_cap: Optional[float] = None
        exposure_cap_cfg = self.cfg.max_notional_rupees
        if exposure_cap_cfg is not None:
            try:
                exposure_cap_val = float(exposure_cap_cfg)
            except (TypeError, ValueError):
                pass
            else:
                if exposure_cap_val >= 0.0:
                    exposure_cap = exposure_cap_val
        cap_abs_setting = float(getattr(settings, "EXPOSURE_CAP_ABS", 0.0) or 0.0)

        allow_min_override = False
        if use_premium_basis:
            available_lots = max(0, self.cfg.max_lots_per_symbol - current_lots)
            quote_payload = quote or {
                "mid": option_mid_price if option_mid_price is not None else entry_price
            }
            runner_for_cap = runner or SimpleNamespace(equity_amount=equity_rupees)
            lots, unit_notional, cap, eq_source = lots_from_premium_cap(
                runner_for_cap,
                quote_payload,
                lot_size,
                available_lots,
            )
            plan["eq_source"] = eq_source
            cap_from_equity = float(cap)
            if exposure_cap is not None:
                cap_from_equity = min(exposure_cap, cap_from_equity)
            exposure_cap = cap_from_equity
            cap = cap_from_equity
            price_mid = float(_mid_from_quote(quote_payload))
            meta: Optional[Dict[str, Any]] = None
            allow_min_config = bool(
                getattr(settings, "risk_allow_min_one_lot", False)
            )
            if (
                lots <= 0
                and allow_min_config
                and available_lots > 0
                and unit_notional > 0
                and float(equity_rupees) >= unit_notional
                and not (cap_abs_setting > 0 and unit_notional > cap_abs_setting)
                and (
                    exposure_cap_cfg is None
                    or float(exposure_cap_cfg) >= unit_notional
                )
            ):
                lots = 1
                allow_min_override = True
                exposure_cap = max(exposure_cap or 0.0, unit_notional)
                cap = exposure_cap
                logger.info(
                    "allow_min_one_lot override: unit_notional=%.2f cap=%.2f",
                    unit_notional,
                    cap,
                )
            if lots <= 0:
                cap_abs_value = (
                    round(cap_abs_setting, 2) if cap_abs_setting > 0 else None
                )
                meta = {
                    "reason": "cap_lt_one_lot",
                    "basis": basis,
                    "price": round(price_mid, 2),
                    "unit_notional": round(unit_notional, 2),
                    "cap": round(cap, 2),
                    "cap_abs": cap_abs_value,
                    "equity_cap_pct": round(
                        float(getattr(settings, "EXPOSURE_CAP_PCT_OF_EQUITY", 0.0)),
                        4,
                    ),
                    "lots": int(lots),
                    "lot_size": int(lot_size),
                    "equity": float(equity_rupees),
                    "eq_source": eq_source,
                    "available_lots": int(available_lots),
                    "current_lots": int(current_lots),
                    "max_lots_per_symbol": int(self.cfg.max_lots_per_symbol),
                }
                if cap_abs_value is None:
                    meta["cap_abs"] = None
            if settings_basis.lower() == "premium" and lots <= 0:
                if meta and meta.get("reason") == "cap_lt_one_lot":
                    logger.info("block_cap_lt_one_lot %s", meta)
                    plan["qty_lots"] = 0
                    plan["reason_block"] = "cap_lt_one_lot"
                    cap_msg = (
                        f"cap_lt_one_lot (cap={cap:.0f}, unit={unit_notional:.0f})"
                    )
                    if cap_abs_setting > 0 and cap_abs_setting <= unit_notional:
                        cap_msg += f" cap_abs={cap_abs_setting:.0f}"
                    reasons = plan.setdefault("reasons", [])
                    if cap_msg not in reasons:
                        reasons.append(cap_msg)
                    return Block(reason="cap_lt_one_lot", details=meta)
            plan["qty_lots"] = int(lots)
            if allow_min_override:
                reasons = plan.setdefault("reasons", [])
                if "allow_min_one_lot" not in reasons:
                    reasons.append("allow_min_one_lot")
            intended_lots = min(intended_lots, int(lots))
        else:
            unit_notional = spot_price * lot_size

        if current_lots + intended_lots > lot_cap:
            return (
                False,
                "lot_cap" if lot_source != "global" else "max_lots_symbol",
                {
                    "sym": intended_symbol,
                    "lots": current_lots,
                    "intended": intended_lots,
                    "cap": lot_cap,
                },
            )
        intended_notional = unit_notional * intended_lots
        if (
            exposure_cap is not None
            and exposure.notional_rupees + intended_notional > exposure_cap
        ):
            logger.info(
                "max_notional block: basis=%s unit_notional=%.2f calc_lots=%d cap=%.2f",
                basis,
                unit_notional,
                intended_lots,
                exposure_cap,
            )
            return (
                False,
                "max_notional",
                {
                    "cur": exposure.notional_rupees,
                    "add": intended_notional,
                },
            )

        gmode = gamma_mode if gamma_mode is not None else self._is_gamma_mode(now)

        if gmode and intended_lots > self.cfg.max_gamma_mode_lots:
            return (
                False,
                "gamma_mode_lot_cap",
                {
                    "intended": intended_lots,
                    "cap": self.cfg.max_gamma_mode_lots,
                },
            )

        cap, cap_source = self._delta_cap_for(regime, gamma_mode=gmode)
        details["delta_cap"] = cap
        if cap_source:
            details["delta_cap_source"] = cap_source
        if cap <= 0:
            return (
                False,
                "delta_cap",
                {"cap": cap, "source": cap_source or "config"},
            )
        if portfolio_delta_units is not None and abs(portfolio_delta_units) > cap:
            return (
                False,
                "delta_cap",
                {
                    "portfolio_delta_units": round(portfolio_delta_units, 1),
                    "cap": cap,
                    "source": cap_source or "config",
                },
            )
        if planned_delta_units is not None and portfolio_delta_units is not None:
            if abs(portfolio_delta_units + planned_delta_units) > cap:
                return (
                    False,
                    "delta_cap_on_add",
                    {
                        "would_be": round(
                            portfolio_delta_units + planned_delta_units, 1
                        ),
                        "cap": cap,
                        "source": cap_source or "config",
                    },
                )

        if len(self.state.roll_R_last10) >= 10:
            avg10 = sum(self.state.roll_R_last10[-10:]) / 10.0
            details["roll10_avgR"] = round(avg10, 3)
            if avg10 < self.cfg.roll10_pause_R:
                self.state.cooloff_until = now + timedelta(
                    minutes=self.cfg.roll10_pause_minutes
                )
                return (
                    False,
                    "roll10_down",
                    {
                        "avg10R": round(avg10, 3),
                        "until": self.state.cooloff_until.isoformat(),
                    },
                )

        if self.state.consecutive_losses >= self.cfg.max_consec_losses:
            self.state.cooloff_until = now + timedelta(minutes=self.cfg.cooloff_minutes)
            return (
                False,
                "loss_cooloff",
                {"until": self.state.cooloff_until.isoformat()},
            )

        if entry_price and stop_loss_price and lot_size and intended_lots:
            r_per_contract = abs(entry_price - stop_loss_price)
            r_rupees = r_per_contract * lot_size * intended_lots
            details["R_rupees_est"] = round(r_rupees, 2)
        if allow_min_override:
            details["allow_min_one_lot"] = True

        return True, "", details

    # -------- POST-TRADE UPDATE --------
    def on_trade_closed(self, *, pnl_R: float, pnl_rupees: float | None = None) -> None:
        now = self._now()
        self.state.new_session_if_needed(now)

        self.state.trades_today += 1
        self.state.cum_R_today += pnl_R
        if pnl_rupees is not None:
            self.state.cum_loss_rupees += pnl_rupees
            self.state.pnl_history_rupees.append(pnl_rupees)
            if len(self.state.pnl_history_rupees) > self._max_var_history:
                self.state.pnl_history_rupees = self.state.pnl_history_rupees[
                    -self._max_var_history :
                ]
        self.state.roll_R_last10.append(pnl_R)
        if len(self.state.roll_R_last10) > 20:
            self.state.roll_R_last10 = self.state.roll_R_last10[-20:]

        if pnl_R < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.state.cum_R_today <= -self.cfg.max_daily_dd_R:
            sd2 = self.state.session_date or ""
            if sd2 not in self.state.daily_caps_hit_recent:
                self.state.daily_caps_hit_recent.append(sd2)

    # -------- UTILITIES --------
    def snapshot(self) -> dict:
        now = self._now()
        self.state.new_session_if_needed(now)
        avg10: Optional[float] = None
        if self.state.roll_R_last10:
            tail = self.state.roll_R_last10[-10:]
            avg10 = round(sum(tail) / len(tail), 3)
        snapshot = {
            "session_date": self.state.session_date,
            "cum_R_today": round(self.state.cum_R_today, 3),
            "cum_loss_rupees": round(self.state.cum_loss_rupees, 2),
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "cooloff_until": (
                self.state.cooloff_until.isoformat()
                if self.state.cooloff_until
                else None
            ),
            "roll10_avgR": avg10,
            "daily_caps_hit_recent": list(self.state.daily_caps_hit_recent),
            "skip_next_open_date": self.state.skip_next_open_date,
        }
        metrics = self._var_metrics()
        if metrics is not None:
            snapshot["var_estimate"] = round(metrics["var"], 2)
            snapshot["cvar_estimate"] = round(metrics["cvar"], 2)
        return snapshot
