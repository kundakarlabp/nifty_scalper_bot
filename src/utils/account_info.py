"""
Account info helpers with graceful fallbacks when Kite creds are absent.
"""

from __future__ import annotations

from typing import Optional

from src.config import settings

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


def get_equity_estimate() -> float:
    """
    Best-effort equity (₹) estimation:
      - Try Kite margins().available.cash; fallback to .net
      - Else fallback to settings.risk.default_equity
    """
    try:
        if not KiteConnect:
            raise RuntimeError("kiteconnect not installed")

        api_key = getattr(settings.zerodha, "api_key", None) or getattr(settings, "ZERODHA_API_KEY", None)
        access_token = getattr(settings.zerodha, "access_token", None) or getattr(settings, "KITE_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("Kite credentials missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        margins = kite.margins(segment="equity")  # dict
        available = margins.get("available") or {}
        raw: Optional[float] = available.get("cash", available.get("net", 0.0))
        return float(raw or 0.0)
    except Exception:
        return float(getattr(settings.risk, "default_equity", getattr(settings, "DEFAULT_EQUITY", 30000.0)))