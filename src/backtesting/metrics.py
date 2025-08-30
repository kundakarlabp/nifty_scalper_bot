from __future__ import annotations

"""Utility metrics for backtest evaluation."""


def reject(summary: dict) -> bool:
    """Return ``True`` if the given summary fails basic thresholds."""

    if not summary or summary.get("trades", 0) < 30:
        return True
    if summary.get("PF", 0) < 1.25:
        return True
    if summary.get("Win%", 0) < 44:
        return True
    if summary.get("AvgR", 0) < 0.20:
        return True
    if summary.get("MaxDD_R", 9) > 6.0:
        return True
    return False
