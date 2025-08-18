# src/risk/session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional

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
    Tracks trades and daily PnL for risk gating.
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

    @property
    def equity(self) -> float:
        return float(self.starting_equity + self.daily_pnl)

    @property
    def daily_drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return float(self.daily_pnl / self.starting_equity)

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

    # ---------- convenience ----------

    def flatten_all(self, exit_price_provider) -> None:
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
