"""Position sizing formulas."""

from __future__ import annotations

from ..config.constants import LOT_SIZE


def kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    fraction: float = 0.25,
    lot_size: int = LOT_SIZE,
) -> int:
    """Kelly fractional position sizing. Returns number of lots."""
    if avg_loss <= 0 or win_rate <= 0:
        return 1
    odds = avg_win / avg_loss
    kelly_pct = win_rate - (1 - win_rate) / odds
    kelly_pct = max(0.0, kelly_pct) * fraction  # fractional Kelly
    capital_to_risk = capital * kelly_pct
    lots = int(capital_to_risk / (avg_loss * lot_size))
    return max(1, lots)


def atr_size(
    capital: float,
    risk_pct: float,
    atr: float,
    lot_size: int = LOT_SIZE,
    tick_size: float = 0.05,
) -> int:
    """ATR-based position sizing."""
    if atr <= 0:
        return 1
    risk_amount = capital * risk_pct / 100.0
    risk_per_lot = atr * lot_size
    lots = int(risk_amount / risk_per_lot)
    return max(1, lots)


def vix_scale(base_lots: int, vix: float, high: float = 20.0, extreme: float = 30.0) -> int:
    """Scale down position size based on VIX level."""
    if vix >= extreme:
        return 0  # Stop trading
    if vix >= high:
        return max(1, base_lots // 2)  # Halve size
    return base_lots
