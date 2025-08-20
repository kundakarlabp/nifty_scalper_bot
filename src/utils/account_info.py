# src/utils/account_info.py
"""
Helpers to query Kite account balance / margins.

This is used by position sizer & risk manager to scale trades
based on actual deployable balance.
"""

from __future__ import annotations

from kiteconnect import KiteConnect
from src.config import Config


def get_dynamic_account_balance() -> float:
    """
    Fetch deployable account balance (cash) from Zerodha Kite.

    Returns:
        float: available cash balance (equity segment).
    """
    kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
    kite.set_access_token(Config.KITE_ACCESS_TOKEN)

    margins = kite.margins(segment="equity")
    try:
        return float(margins["equity"]["available"]["cash"])
    except Exception:
        # Fallback: return 0.0 if structure unexpected
        return 0.0
