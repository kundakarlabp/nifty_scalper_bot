# src/risk/session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Callable

from src.config import RiskConfig, ExecutorConfig


@dataclass
class Trade:
    """
    Minimal trade bookkeeping for session-level risk gates.
    """
    timestamp_open: datetime
    side: str  # "BUY" or "SELL"
    symbol: str
    qty: int
    entry_price: float
    stop_loss: Optional[float] = None
    target: Optional[float] = None

    # Runtime fields
    timestamp_close: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0

    def close(self, price: float, when: Optional[datetime] = None) -> None:
        self.timestamp_close = when or datetime.now()
        self.exit_price = float(price)
        sign = 1.0 if self.side.upper() == "BUY" else -1.0
        self.pnl = (float(self.exit_price) - float(self.entry_price)) * float(self.qty) * sign


@dataclass
class TradingSession:
    """
    Tracks trades and daily PnL for risk gating and basic session controls.
    The gating thresholds come from RiskConfig; all reads use getattr with safe defaults.
    """
    risk_cfg: RiskConfig
    exec_cfg: ExecutorConfig
    starting_equity: float = 100_000.0

    start_day: date = field(default_factory=lambda: datetime.now().date())
    active_trades: List[Trade] = field(default_factory=list)
    trade_history: List[Trade] = field(default_factory=list)
    trades_today: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0

    # session controls
    _pause_until: Optional[datetime] = None
    _day_stopped: bool = False
    _started_at: datetime = field(default_factory=datetime.now)

    # ---------- properties ----------

    @property
    def equity(self) -> float:
        return float(self.starting_equity + self.daily_pnl)

    @property
    def daily_drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return float(self.daily_pnl / self.starting_equity)

    @property
    def uptime_sec(self) -> float:
        return max(0.0, (datetime.now() - self._started_at).total_seconds())

    @property
    def paused(self) -> bool:
        return bool(self._pause_until and datetime.now() < self._pause_until)

    # ---------- lifecycle ----------

    def _rollover_if_new_day(self) -> None:
        today = datetime.now().date()
        if today != self.start_day:
            # Reset daily counters
            self.start_day = today
            self.trade_history.clear()
            self.active_trades.clear()
            self.trades_today = 0
            self.consecutive_losses = 0
            self.daily_pnl = 0.0
            self._pause_until = None
            self._day_stopped = False
            self._started_at = datetime.now()

    # ---------- gating helpers ----------

    def _max_daily_dd_pct(self) -> float:
        # e.g. MAX_DAILY_DRAWDOWN_PCT in .env (0.05 = 5%)
        return float(getattr(self.risk_cfg, "max_daily_drawdown_pct", 0.05) or 0.05)

    def _loss_streak_limit(self) -> int:
        return int(getattr(self.risk_cfg, "consecutive_loss_limit", 3) or 3)

    def _max_trades_per_day(self) -> int:
        return int(getattr(self.risk_cfg, "max_trades_per_day", 30) or 30)

    def _max_concurrent_positions(self) -> int:
        return int(getattr(self.risk_cfg, "max_concurrent_positions", 1) or 1)

    def _halt_on_drawdown(self) -> bool:
        return bool(getattr(self.risk_cfg, "halt_on_drawdown", True))

    def is_day_stopped(self) -> bool:
        """True if session has been halted for the day due to risk triggers."""
        if self._day_stopped:
            return True
        # Drawdown gate
        if self._halt_on_drawdown() and self.daily_drawdown_pct <= -abs(self._max_daily_dd_pct()):
            return True
        # Loss streak
        if self.consecutive_losses >= self._loss_streak_limit():
            return True
        # Trades/day cap
        if self.trades_today >= self._max_trades_per_day():
            return True
        return False

    def should_allow_new_entry(self) -> bool:
        self._rollover_if_new_day()
        if self.paused:
            return False
        if self.is_day_stopped():
            self._day_stopped = True
            return False
        if len(self.active_trades) >= self._max_concurrent_positions():
            return False
        return True

    # ---------- controls ----------

    def pause_entries(self, minutes: int = 1) -> None:
        minutes = max(1, int(minutes))
        self._pause_until = datetime.now() + timedelta(minutes=minutes)

    def resume_entries(self) -> None:
        self._pause_until = None

    def stop_for_day(self) -> None:
        self._day_stopped = True

    # ---------- trade events ----------

    def on_order_filled_open(
        self,
        side: str,
        symbol: str,
        qty: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
    ) -> Trade:
        self._rollover_if_new_day()
        t = Trade(
            timestamp_open=datetime.now(),
            side=side.upper(),
            symbol=symbol,
            qty=int(qty),
            entry_price=float(entry_price),
            stop_loss=stop_loss,
            target=target,
        )
        self.active_trades.append(t)
        self.trades_today += 1
        return t

    def on_order_filled_close(self, trade: Trade, exit_price: float) -> None:
        trade.close(exit_price)
        self.daily_pnl += float(trade.pnl)
        self.trade_history.append(trade)
        if trade in self.active_trades:
            self.active_trades.remove(trade)

        # Loss streak logic (per closed trade)
        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # If a close pushed us below daily DD limit, lock the day
        if self._halt_on_drawdown() and self.daily_drawdown_pct <= -abs(self._max_daily_dd_pct()):
            self._day_stopped = True

    # ---------- convenience ----------

    def flatten_all(self, exit_price_provider: Callable[[str], float]) -> None:
        """
        Best-effort flatten: exit all open trades using a provided price function
        (symbol -> exit_price).
        """
        for t in list(self.active_trades):
            try:
                px = float(exit_price_provider(t.symbol))
            except Exception:
                px = float(t.entry_price)  # worst case
            self.on_order_filled_close(t, px)
