"""Central risk limits engine handling pre and post trade checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


@dataclass
class LimitConfig:
    """Configuration knobs for :class:`RiskEngine`."""

    tz: str = "Asia/Kolkata"
    max_daily_dd_R: float = 2.5
    max_trades_per_session: int = 40
    max_lots_per_symbol: int = 5
    max_notional_rupees: float = 1_500_000.0
    max_gamma_mode_lots: int = 2
    max_portfolio_delta_units: int = 100
    max_portfolio_delta_units_gamma: int = 60
    roll10_pause_R: float = -0.2
    roll10_pause_minutes: int = 60
    cooloff_losses: int = 3
    cooloff_minutes: int = 45
    skip_next_open_after_two_daily_caps: bool = True


@dataclass
class Exposure:
    """Snapshot of current market exposure."""

    lots_by_symbol: Dict[str, int] = field(default_factory=dict)
    notional_rupees: float = 0.0


@dataclass
class RiskState:
    """Mutable state tracked by :class:`RiskEngine`."""

    tz: str = "Asia/Kolkata"
    session_date: Optional[str] = None
    cum_R_today: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    roll_R_last10: List[float] = field(default_factory=list)
    cooloff_until: Optional[datetime] = None
    daily_caps_hit_recent: List[str] = field(default_factory=list)
    skip_next_open_date: Optional[str] = None

    def new_session_if_needed(self, now_ist: datetime) -> None:
        """Reset per-session counters if the date changed."""

        d = now_ist.date().isoformat()
        if self.session_date != d:
            self.session_date = d
            self.cum_R_today = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.cooloff_until = None


class RiskEngine:
    """Risk gatekeeper performing pre-trade checks and post-trade updates."""

    def __init__(self, cfg: LimitConfig) -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.tz)
        self.state = RiskState(cfg.tz)

    # -------- helpers --------
    def _now(self) -> datetime:
        return datetime.now(self.tz)

    def _is_gamma_mode(self, now: datetime) -> bool:
        # Tuesday (1) at or after 14:45 IST
        return now.weekday() == 1 and (now.time() >= time(14, 45))

    # -------- PRE-TRADE CHECK --------
    def pre_trade_check(
        self,
        *,
        equity_rupees: float,
        plan: dict,
        exposure: Exposure,
        intended_symbol: str,
        intended_lots: int,
        lot_size: int,
        entry_price: float,
        stop_loss_price: float,
        planned_delta_units: Optional[float] = None,
        portfolio_delta_units: Optional[float] = None,
        gamma_mode: Optional[bool] = None,
    ) -> Tuple[bool, str, Dict]:
        """Return (ok, reason_block, details) for a proposed trade."""

        now = self._now()
        self.state.new_session_if_needed(now)

        details: Dict[str, float] = {}

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

        if self.state.trades_today >= self.cfg.max_trades_per_session:
            return (
                False,
                "max_trades_session",
                {"trades_today": self.state.trades_today},
            )

        current_lots = exposure.lots_by_symbol.get(intended_symbol, 0)
        if current_lots + intended_lots > self.cfg.max_lots_per_symbol:
            return (
                False,
                "max_lots_symbol",
                {
                    "sym": intended_symbol,
                    "lots": current_lots,
                    "intended": intended_lots,
                },
            )

        intended_notional = entry_price * lot_size * intended_lots
        if exposure.notional_rupees + intended_notional > self.cfg.max_notional_rupees:
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

        cap = (
            self.cfg.max_portfolio_delta_units_gamma
            if gmode
            else self.cfg.max_portfolio_delta_units
        )
        if portfolio_delta_units is not None and abs(portfolio_delta_units) > cap:
            return (
                False,
                "delta_cap",
                {
                    "portfolio_delta_units": round(portfolio_delta_units, 1),
                    "cap": cap,
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

        if self.state.consecutive_losses >= self.cfg.cooloff_losses:
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

        return True, "", details

    # -------- POST-TRADE UPDATE --------
    def on_trade_closed(self, *, pnl_R: float) -> None:
        now = self._now()
        self.state.new_session_if_needed(now)

        self.state.trades_today += 1
        self.state.cum_R_today += pnl_R
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
        return {
            "session_date": self.state.session_date,
            "cum_R_today": round(self.state.cum_R_today, 3),
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
