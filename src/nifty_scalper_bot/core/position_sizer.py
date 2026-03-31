"""Position sizing helpers with Kelly-style scaling."""

from __future__ import annotations


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_kelly_fraction: float = 0.25,
) -> float:
    """Compute Kelly fraction. Args: win_rate,avg_win,avg_loss,max_kelly_fraction. Returns: float. Raises: None."""

    p = max(0.0, min(float(win_rate), 1.0))
    loss = max(float(avg_loss), 1e-9)
    b = max(float(avg_win), 0.0) / loss
    if b <= 0:
        return 0.0
    kelly_fraction = p - ((1.0 - p) / b)
    if p < 0.4:
        kelly_fraction *= 0.5
    return max(0.0, min(float(max_kelly_fraction), kelly_fraction))


def size_quantity(
    base_qty: int,
    regime_scale: float,
    confidence: float,
    kelly_fraction: float,
    capital_fraction: float,
    lot_size: int = 1,
    max_capital_exposure_qty: int | None = None,
) -> int:
    """Compute final qty. Args: base_qty,regime_scale,confidence,kelly_fraction,capital_fraction,lot_size,max_capital_exposure_qty. Returns: int. Raises: None."""

    qty = (
        float(base_qty)
        * max(float(regime_scale), 0.0)
        * max(float(confidence), 0.0)
        * max(float(kelly_fraction), 0.0)
        * max(float(capital_fraction), 0.0)
    )
    rounded = max(int(round(qty)), int(lot_size))
    if max_capital_exposure_qty is not None:
        rounded = min(rounded, max(1, int(max_capital_exposure_qty)))
    return max(rounded, int(lot_size))
