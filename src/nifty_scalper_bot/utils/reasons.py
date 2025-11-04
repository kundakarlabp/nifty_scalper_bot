"""Canonical reason code helpers for risk and session reporting."""

from __future__ import annotations

SOFT: set[str] = {
    "COOLDOWN",
    "STALE",
    "RECENT_REJECT",
    "RISK_STATE",
    "TRADING_DISABLED",
    "MARGIN",
    "MIS_WINDOW_CLOSED",
    "AMO_ONLY",
    "MARKET_CLOSED",
}
HARD: set[str] = {"BREAKER"}

KNOWN: dict[str, str] = {
    "NO_QUOTE_DATA": "no_quote_data",
    "no_quote_data": "no_quote_data",
    "NO_REFERENCE_PRICE": "no_reference_price",
    "no_reference_price": "no_reference_price",
    "MARGIN_NO_QTY": "margin_no_qty",
    "margin_no_qty": "margin_no_qty",
}


def canonical(reason: str | None) -> str:
    """Return standardized skip reason token for *reason* input.

    Args:
        reason: Free-form reason text supplied by risk or broker layers.

    Returns:
        str: Canonical reason token suitable for metrics and telemetry.

    Raises:
        None.
    """

    token_raw = (reason or "").strip()
    if not token_raw:
        return "OK"
    if token_raw in KNOWN:
        return KNOWN[token_raw]
    token_upper = token_raw.upper()
    if token_upper in KNOWN:
        return KNOWN[token_upper]
    if "MARGIN_NO_QTY" in token_upper:
        return "margin_no_qty"
    if "MARGIN_COOLDOWN" in token_upper:
        return "margin_cooldown"
    if "INSUFFICIENT FUNDS" in token_upper or "MARGIN" in token_upper:
        return "MARGIN"
    if "MIS" in token_upper and "WINDOW" in token_upper:
        return "MIS_WINDOW_CLOSED"
    if "INTRADAY ORDERS (MIS) ARE ALLOWED ONLY" in token_upper:
        return "MIS_WINDOW_CLOSED"
    if "AMO" in token_upper and "ONLY" in token_upper:
        return "AMO_ONLY"
    if "MARKET" in token_upper and "CLOSED" in token_upper:
        return "MARKET_CLOSED"
    if "COOLDOWN" in token_upper:
        return "COOLDOWN"
    if "STALE" in token_upper:
        return "STALE"
    if "RISK_STATE" in token_upper or "TRADING DISABLED BY RISK STATE" in token_upper:
        return "RISK_STATE"
    if "TRADING_DISABLED" in token_upper or "TRADING DISABLED" in token_upper:
        return "TRADING_DISABLED"
    if "BREAKER" in token_upper:
        return "BREAKER"
    return token_upper


__all__ = ["canonical", "SOFT", "HARD", "KNOWN"]
