# src/utils/account_info.py
"""
Account info helpers (graceful fallbacks).

- get_equity_estimate(kite) -> float
    Returns a conservative equity estimate using Zerodha margins when available,
    else falls back to settings.risk.default_equity.

Notes:
- Narrow exception handling; no blanket 'except Exception'.
- Does not crash if kite is None or creds are missing.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import settings

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    class NetworkException(Exception): ...  # type: ignore
    class TokenException(Exception): ...    # type: ignore

logger = logging.getLogger(__name__)


def get_equity_estimate(kite: Optional["KiteConnect"]) -> float:
    """
    Best-effort equity figure to drive sizing logic.
    Priority:
      1) Zerodha 'margins()' API (equity.cash / available)
      2) settings.risk.default_equity
    """
    default_equity = float(getattr(settings.risk, "default_equity", 30000.0))

    if kite is None:
        logger.info("account_info: no Kite client; using default equity %.2f", default_equity)
        return default_equity

    try:
        m = kite.margins()  # may raise on network/token issues
        # Typical structure: {'equity': {'available': {...}, 'net': ..., ...}, ...}
        eq = m.get("equity") or {}
        # Use 'available' cash first, then 'net'
        avail = eq.get("available") or {}
        # Zerodha uses keys like 'cash', 'adhoc_margin', etc.
        cash = avail.get("cash")
        if cash is not None:
            return float(cash)
        net = eq.get("net")
        if net is not None:
            return float(net)
        # As a last resort, try top-level 'available' if present
        if isinstance(m.get("available"), (int, float, str)):
            return float(m["available"])
        logger.warning("account_info: margins() returned unexpected shape; using default equity.")
        return default_equity
    except (NetworkException, TokenException) as e:
        logger.warning("account_info: margins() failed (%s); using default equity.", e)
        return default_equity
    except (TypeError, ValueError) as e:
        logger.warning("account_info: parsing margins() failed (%s); using default equity.", e)
        return default_equity
