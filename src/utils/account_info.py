# src/utils/account_info.py
# Optional import to avoid hard crash when kiteconnect is absent
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

from src.config import Config

def get_dynamic_account_balance() -> float:
    if KiteConnect is None:
        raise ImportError("kiteconnect is not installed. pip install kiteconnect")
    kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
    access = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
    if not access:
        raise RuntimeError("Access token not configured in Config.")
    kite.set_access_token(access)
    margins = kite.margins(segment="equity")
    # Use 'net' to reflect collateral/used margins if preferred
    return float(margins.get('available', {}).get('cash', 0.0))
