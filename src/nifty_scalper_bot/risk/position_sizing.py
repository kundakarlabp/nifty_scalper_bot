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
    volatility: float = 0.0


class RiskManager:
    """Hard risk blocker for strategy execution."""

    def __init__(
        self,
        max_daily_loss: float,
        max_trades: int,
        max_open_exposure: float,
        min_volatility: float = 0.0,
    ) -> None:
        """Initialise limits. Args: limits. Returns: None. Raises: ValueError."""
        self.max_daily_loss = abs(float(max_daily_loss))
        self.max_trades = int(max_trades)
        self.max_open_exposure = float(max_open_exposure)
        self.min_volatility = float(min_volatility)

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
        if snapshot.volatility < self.min_volatility:
            reason = "volatility_filter"
            logger.warning('{"event":"RISK_BLOCKED_TRADE","reason":"%s"}', reason)
            return False, reason
        return True, None


@dataclass(slots=True)
class PositionSizingResult:
    allowed: bool
    qty: int
    reason: str


class PositionSizer:
    """Risk-based lot sizing."""

    def size(self, *, equity: float, risk_per_trade_pct: float, entry: float, stop_loss: float, lot_size: int, confidence: float, regime_multiplier: float, max_lots: int, available_margin: float, margin_per_lot: float) -> PositionSizingResult:
        if entry <= stop_loss:
            return PositionSizingResult(False, 0, 'invalid_stop_loss')
        per_unit_risk = entry - stop_loss
        if per_unit_risk <= 0:
            return PositionSizingResult(False, 0, 'invalid_per_unit_risk')
        risk_amount = equity * risk_per_trade_pct * max(0.5, min(1.2, confidence)) * max(0.5, regime_multiplier)
        raw_qty = int(risk_amount // per_unit_risk)
        qty = (raw_qty // lot_size) * lot_size
        cap_lots = 1 if equity <= 30000 and confidence < 0.9 else max_lots
        qty = min(qty, cap_lots * lot_size)
        max_margin_qty = int(available_margin // max(margin_per_lot, 1.0)) * lot_size
        qty = min(qty, max_margin_qty)
        if qty < lot_size:
            if raw_qty < lot_size:
                return PositionSizingResult(False, 0, 'insufficient_risk_budget_for_one_lot')
            return PositionSizingResult(False, 0, 'insufficient_qty_or_margin')
        logger.info('POSITION_SIZE_DECISION equity=%s risk_amount=%s entry=%s sl=%s qty=%s', equity, risk_amount, entry, stop_loss, qty)
        return PositionSizingResult(True, qty, 'ok')
