# src/utils/account_info.py
from __future__ import annotations

from typing import Optional
from kiteconnect import KiteConnect

from src.config import settings


def get_dynamic_account_balance() -> float:
    """
    Returns ₹ cash available (falls back to 'net' if 'cash' isn't present).
    """
    kite = KiteConnect(api_key=settings.ZERODHA_API_KEY)
    kite.set_access_token(settings.KITE_ACCESS_TOKEN)
    margins = kite.margins(segment="equity")  # dict from Kite

    available = margins.get("available") or {}
    raw: Optional[float] = available.get("cash", available.get("net", 0.0))
    try:
        return float(raw or 0.0)
    except Exception:
        return 0.0
