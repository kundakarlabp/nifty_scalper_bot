"""Signal-level helper utilities and runtime patches."""

from __future__ import annotations

def check_atr_band(atr_pct: float, min_val: float, max_val: float) -> tuple[bool, str | None]:
    """Return whether ``atr_pct`` falls within the inclusive ``[min_val, max_val]`` band.

    Parameters
    ----------
    atr_pct:
        Observed ATR percentage of the underlying instrument.
    min_val:
        Lower bound for the acceptable ATR percentage.
    max_val:
        Upper bound for the acceptable ATR percentage.

    Returns
    -------
    Tuple[bool, str | None]
        ``(True, None)`` when within the configured band, otherwise ``(False, reason)``
        where ``reason`` contains a diagnostic message including the offending value
        and configured limits.
    """

    if atr_pct < min_val:
        return False, f"atr_out_of_band: atr={atr_pct:.4f} < min={min_val}"
    if atr_pct > max_val:
        return False, f"atr_out_of_band: atr={atr_pct:.4f} > max={max_val}"
    return True, None


__all__ = ["check_atr_band"]
