# src/utils/account_info.py
from __future__ import annotations

import logging
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


def get_equity_estimate(kite_instance: Optional[object] = None) -> float:
    """
    Try to read available cash from Kite funds; fall back to .env default.
    Never returns NaN/negative; returns 0.0 only if everything fails.
    """
    # 1) Broker funds
    if kite_instance is not None:
        try:
            funds = kite_instance.margins("equity")  # type: ignore[attr-defined]
            # Zerodha response often has: net, available: { cash, adhoc_margin, ... }
            if isinstance(funds, dict):
                # Prefer immediately usable cash; else net
                avail = funds.get("available") or {}
                cash = float(avail.get("cash", 0.0) or 0.0)
                if cash > 0:
                    return cash
                net = float(funds.get("net", 0.0) or 0.0)
                if net > 0:
                    return net
        except Exception as e:
            log.debug("get_equity_estimate: funds fetch failed: %s", e)

    # 2) Fallback to configured default
    try:
        eq = float(getattr(settings.risk, "default_equity", 0.0) or 0.0)
        return max(0.0, eq)
    except Exception:
        return 0.0
