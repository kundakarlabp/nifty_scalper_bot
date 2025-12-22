"""Comprehensive risk manager for validating trades."""

from __future__ import annotations
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Deque, Literal, Protocol, Sequence
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.base import RiskConfig

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
MARKET_BUFFER = timedelta(minutes=5)
BROKER_ERROR_THRESHOLD = 5
BROKER_ERROR_WINDOW = timedelta(minutes=1)


class PositionLike(Protocol):
    """Protocol describing the shape of an open position."""

    symbol: str
    side: Literal["LONG", "SHORT"]
    quantity: int
    entry_price: float
    stop_loss: float | None
    take_profit: float | None


class PositionManager(Protocol):
    """Protocol for the position manager dependency."""

    def get_open_positions(self) -> Sequence[PositionLike]:
        """Return all currently open positions."""

    def get_position(self, symbol: str) -> PositionLike | None:
        """Return an open position for *symbol* if one exists."""


@dataclass
class RiskManager:
    """Validate all trades against risk parameters."""

    risk_config: RiskConfig
    position_manager: PositionManager
    initial_balance: float

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        self._account_balance: float = self.initial_balance
        self._peak_balance: float = self.initial_balance
        self._start_of_day_balance: float = self.initial_balance
        self._daily_realized_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._consecutive_losses: int = 0
        self._circuit_breaker_tripped: bool = False
        self._circuit_breaker_reason: str = ""
        self._circuit_breaker_timestamp: datetime | None = None
        self._broker_error_timestamps: Deque[datetime] = deque()
        self._last_reset_date = self._now().date()

    # ------------------------------------------------------------------
    # Public API
    def validate_new_position(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        quantity: int,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> tuple[bool, str]:
        """Validate if new position can be opened."""

        self._auto_reset_daily_counters()
        can_trade, reason = self.can_trade_now()
        if not can_trade:
            return False, reason

        if quantity <= 0:
            return False, "Quantity must be positive"

        if entry_price <= 0:
            return False, "Entry price must be positive"

        side_normalized = side.upper()
        if side_normalized not in {"LONG", "SHORT"}:
            return False, "Invalid side"

        if side_normalized == "SHORT" and not self.risk_config.allow_short:
            return False, "Short selling is disabled"

        open_positions = self._get_open_positions()
        if len(open_positions) >= self.risk_config.max_concurrent_positions:
            return False, "Maximum concurrent positions reached"

        symbol_positions = self._count_positions_for_symbol(open_positions, symbol)
        if symbol_positions >= self.risk_config.max_positions_per_symbol:
            return False, "Maximum positions for symbol reached"

        notional = quantity * entry_price
        max_position_notional = (
            self._account_balance * self.risk_config.max_position_size_pct / 100.0
        )
        if notional > max_position_notional:
            return False, "Position size exceeds limit"

        total_exposure = self._calculate_total_exposure(open_positions)
        max_total_exposure = (
            self._account_balance * self.risk_config.max_total_exposure_pct / 100.0
        )
        if total_exposure + notional > max_total_exposure:
            return False, "Total exposure limit exceeded"

        if stop_loss is None or take_profit is None:
            return False, "Stop loss and take profit are required"

        risk_value, reward_value = self._calculate_risk_reward(
            side_normalized, entry_price, stop_loss, take_profit
        )
        if risk_value <= 0:
            return False, "Invalid stop loss distance"

        if reward_value <= 0:
            return False, "Invalid take profit distance"

        ratio = reward_value / risk_value
        if ratio < self.risk_config.min_risk_reward_ratio:
            return False, "Risk-reward ratio below minimum"

        if self._daily_trade_count >= self.risk_config.max_daily_trades:
            return False, "Daily trade limit reached"

        self._daily_trade_count += 1
        return True, ""

    def validate_close_position(
        self,
        symbol: str,
        exit_price: float,
    ) -> tuple[bool, str]:
        """Validate if position can be closed."""

        if exit_price <= 0:
            return False, "Exit price must be positive"

        position = self._get_position(symbol)
        if position is None:
            return False, "No open position for symbol"

        return True, ""

    def update_account_balance(self, realized_pnl: float) -> None:
        """Update account balance with realized P&L."""

        self._auto_reset_daily_counters()
        self._account_balance += realized_pnl
        self._daily_realized_pnl += realized_pnl
        self._peak_balance = max(self._peak_balance, self._account_balance)

        if realized_pnl < 0:
            self._consecutive_losses += 1
        elif realized_pnl > 0:
            self._consecutive_losses = 0

        if self._consecutive_losses >= 3:
            self.trip_circuit_breaker("Three consecutive losing trades")

        drawdown_ok, drawdown_pct = self.check_drawdown()
        if not drawdown_ok:
            self.trip_circuit_breaker(
                (
                    "Drawdown limit exceeded "
                    f"({drawdown_pct:.2f}% > "
                    f"{self.risk_config.max_drawdown_pct:.2f}%)"
                )
            )

        loss_ok, today_loss = self.check_daily_loss()
        if not loss_ok:
            allowed_loss_limit = (
                self._start_of_day_balance * self.risk_config.max_daily_loss_pct / 100.0
            )
            self.trip_circuit_breaker(
                "Daily loss limit exceeded ("
                f"{today_loss:.2f} > "
                f"{allowed_loss_limit:.2f})"
            )

    def check_drawdown(self) -> tuple[bool, float]:
        """Check if drawdown is within limits."""

        if self._peak_balance <= 0:
            return True, 0.0
        drawdown = max(self._peak_balance - self._account_balance, 0.0)
        drawdown_pct = (drawdown / self._peak_balance) * 100.0
        return drawdown_pct <= self.risk_config.max_drawdown_pct, drawdown_pct

    def check_daily_loss(self) -> tuple[bool, float]:
        """Check if daily loss is within limits."""

        today_loss = max(-self._daily_realized_pnl, 0.0)
        allowed_loss = self._start_of_day_balance * (
            self.risk_config.max_daily_loss_pct / 100.0
        )
        return today_loss <= allowed_loss, today_loss

    def is_market_open(self) -> bool:
        """Check if market is currently open (9:15 AM - 3:30 PM IST)."""

        now = self._now()
        if now.weekday() >= 5:  # Saturday/Sunday
            return False

        start_dt = datetime.combine(now.date(), MARKET_OPEN, IST) + MARKET_BUFFER
        end_dt = datetime.combine(now.date(), MARKET_CLOSE, IST) - MARKET_BUFFER

        return start_dt <= now <= end_dt

    def can_trade_now(self) -> tuple[bool, str]:
        """Check if trading is allowed right now."""

        self._auto_reset_daily_counters()

        if self._circuit_breaker_tripped:
            return False, self._circuit_breaker_reason or "Circuit breaker active"

        if not self.is_market_open():
            return False, "Market is closed"

        drawdown_ok, drawdown_pct = self.check_drawdown()
        if not drawdown_ok:
            self.trip_circuit_breaker(
                (
                    "Drawdown limit exceeded "
                    f"({drawdown_pct:.2f}% > "
                    f"{self.risk_config.max_drawdown_pct:.2f}%)"
                )
            )
            return False, self._circuit_breaker_reason

        loss_ok, today_loss = self.check_daily_loss()
        if not loss_ok:
            allowed_loss_limit = (
                self._start_of_day_balance * self.risk_config.max_daily_loss_pct / 100.0
            )
            self.trip_circuit_breaker(
                "Daily loss limit exceeded ("
                f"{today_loss:.2f} > "
                f"{allowed_loss_limit:.2f})"
            )
            return False, self._circuit_breaker_reason

        if self._daily_trade_count >= self.risk_config.max_daily_trades:
            return False, "Daily trade limit reached"

        return True, ""

    def get_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 1.0,
    ) -> int:
        """Calculate optimal position size based on risk percentage."""

        if entry_price <= 0 or stop_loss <= 0:
            raise ValueError("Prices must be positive")

        if risk_pct <= 0:
            raise ValueError("Risk percentage must be positive")

        open_position = self._get_position(symbol)
        if open_position is not None:
            side = open_position.side
        else:
            side = "LONG" if stop_loss < entry_price else "SHORT"

        stop_distance = self._calculate_stop_distance(side, entry_price, stop_loss)

        if stop_distance <= 0:
            return 0

        risk_amount = self._account_balance * (risk_pct / 100.0)
        if risk_amount <= 0:
            return 0

        position_size = int(risk_amount // stop_distance)

        max_position_notional = (
            self._account_balance * self.risk_config.max_position_size_pct / 100.0
        )
        if entry_price > 0:
            max_size_by_notional = int(max_position_notional // entry_price)
            if max_size_by_notional > 0:
                position_size = min(position_size, max_size_by_notional)

        return max(position_size, 0)

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at market open)."""

        now = self._now()
        self._daily_realized_pnl = 0.0
        self._daily_trade_count = 0
        self._consecutive_losses = 0
        self._start_of_day_balance = self._account_balance
        self._last_reset_date = now.date()
        self._broker_error_timestamps.clear()

    def trip_circuit_breaker(self, reason: str) -> None:
        """Trip circuit breaker and disable trading."""

        if not self._circuit_breaker_tripped:
            self._circuit_breaker_tripped = True
            self._circuit_breaker_reason = reason
            self._circuit_breaker_timestamp = self._now()

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (admin only)."""

        self._circuit_breaker_tripped = False
        self._circuit_breaker_reason = ""
        self._circuit_breaker_timestamp = None
        self._consecutive_losses = 0
        self._broker_error_timestamps.clear()

    def is_circuit_breaker_tripped(self) -> tuple[bool, str]:
        """Check circuit breaker status."""

        return self._circuit_breaker_tripped, self._circuit_breaker_reason

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics for monitoring."""

        self._prune_broker_errors()
        open_positions = self._get_open_positions()
        _, drawdown_pct = self.check_drawdown()
        _, today_loss = self.check_daily_loss()
        return {
            "account_balance": self._account_balance,
            "peak_balance": self._peak_balance,
            "start_of_day_balance": self._start_of_day_balance,
            "daily_realized_pnl": self._daily_realized_pnl,
            "daily_loss": today_loss,
            "drawdown_pct": drawdown_pct,
            "open_positions": len(open_positions),
            "total_exposure": self._calculate_total_exposure(open_positions),
            "daily_trade_count": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "circuit_breaker": self._circuit_breaker_tripped,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "broker_errors_last_minute": len(self._broker_error_timestamps),
        }

    def is_circuit_breaker_active(self) -> bool:
        """Return whether the circuit breaker is currently active."""

        return self._circuit_breaker_tripped

    def record_broker_error(self) -> None:
        """Record a broker error and trigger the circuit breaker if needed."""

        now = self._now()
        self._broker_error_timestamps.append(now)
        self._prune_broker_errors(now)

        if len(self._broker_error_timestamps) >= BROKER_ERROR_THRESHOLD:
            self.trip_circuit_breaker("Broker error threshold exceeded")

    # ------------------------------------------------------------------
    # Internal helpers
    def _auto_reset_daily_counters(self) -> None:
        now = self._now()
        if now.date() != self._last_reset_date:
            self.reset_daily_counters()

    def _get_open_positions(self) -> Sequence[PositionLike]:
        getter = getattr(self.position_manager, "get_open_positions", None)
        if callable(getter):
            positions = getter()
            return list(positions)
        return []

    def _get_position(self, symbol: str) -> PositionLike | None:
        getter = getattr(self.position_manager, "get_position", None)
        if callable(getter):
            return getter(symbol)
        for position in self._get_open_positions():
            if position.symbol == symbol:
                return position
        return None

    def _count_positions_for_symbol(
        self, positions: Sequence[PositionLike], symbol: str
    ) -> int:
        return sum(1 for position in positions if position.symbol == symbol)

    def _calculate_total_exposure(self, positions: Sequence[PositionLike]) -> float:
        exposure = 0.0
        for position in positions:
            exposure += abs(position.quantity * position.entry_price)
        return exposure

    def _calculate_risk_reward(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[float, float]:
        stop_distance = self._calculate_stop_distance(side, entry_price, stop_loss)
        reward_distance = self._calculate_reward_distance(
            side, entry_price, take_profit
        )
        return stop_distance, reward_distance

    def _calculate_stop_distance(
        self, side: str, entry_price: float, stop_loss: float
    ) -> float:
        if side == "LONG":
            return entry_price - stop_loss
        return stop_loss - entry_price

    def _calculate_reward_distance(
        self, side: str, entry_price: float, take_profit: float
    ) -> float:
        if side == "LONG":
            return take_profit - entry_price
        return entry_price - take_profit

    def _now(self) -> datetime:
        return datetime.now(IST)

    def _prune_broker_errors(self, now: datetime | None = None) -> None:
        reference = now or self._now()
        cutoff = reference - BROKER_ERROR_WINDOW
        while (
            self._broker_error_timestamps and self._broker_error_timestamps[0] < cutoff
        ):
            self._broker_error_timestamps.popleft()


__all__ = [
    "RiskManager",
    "PositionLike",
    "PositionManager",
]
