"""Deterministic risk and position sizing guards."""

from __future__ import annotations

from dataclasses import dataclass

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RiskSnapshot:
    """Runtime risk state. Args: fields. Returns: snapshot. Raises: None."""

    day_pnl: float
    trades_today: int
    open_exposure: float


class RiskManager:
    """Hard risk blocker for strategy execution."""

    def __init__(
        self,
        max_daily_loss: float,
        max_trades: int,
        max_open_exposure: float,
    ) -> None:
        """Initialise limits. Args: limits. Returns: None. Raises: ValueError."""
        self.max_daily_loss = abs(float(max_daily_loss))
        self.max_trades = int(max_trades)
        self.max_open_exposure = float(max_open_exposure)

    def allow_trade(self, snapshot: RiskSnapshot) -> tuple[bool, str | None]:
        """Validate trade permission. Args: snapshot. Returns: allow/reason. Raises: None."""
        if snapshot.day_pnl <= -self.max_daily_loss:
            reason = "daily_loss_limit"
            logger.warning('{"event":"RISK_BLOCKED_TRADE","reason":"%s"}', reason)
            return False, reason
        if snapshot.trades_today >= self.max_trades:
            reason = "max_trades_reached"
            logger.warning('{"event":"RISK_BLOCKED_TRADE","reason":"%s"}', reason)
            return False, reason
        if snapshot.open_exposure >= self.max_open_exposure:
            reason = "open_exposure_limit"
            logger.warning('{"event":"RISK_BLOCKED_TRADE","reason":"%s"}', reason)
            return False, reason
        return True, None
