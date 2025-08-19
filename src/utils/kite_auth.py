# src/utils/kite_auth.py
from __future__ import annotations
import os
import logging

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import TokenException  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore
    TokenException = Exception  # type: ignore

log = logging.getLogger(__name__)

REQUIRED_MSG = (
    "Provide either ZERODHA_ACCESS_TOKEN, or both ZERODHA_API_SECRET and ZERODHA_REQUEST_TOKEN."
)

def build_kite_from_env() -> "KiteConnect":
    """
    Returns an authenticated KiteConnect client using env vars:
    - ZERODHA_API_KEY (required)
    - EITHER ZERODHA_ACCESS_TOKEN OR (ZERODHA_API_SECRET + ZERODHA_REQUEST_TOKEN)
    Verifies session by calling profile().
    """
    if KiteConnect is None:
        raise RuntimeError("kiteconnect not installed")

    api_key = os.getenv("ZERODHA_API_KEY")
    if not api_key:
        raise RuntimeError("ZERODHA_API_KEY missing")

    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    request_token = os.getenv("ZERODHA_REQUEST_TOKEN")

    kite = KiteConnect(api_key=api_key)

    if access_token:
        kite.set_access_token(access_token)
        try:
            kite.profile()  # quick validation
            log.info("Kite session OK via ACCESS_TOKEN")
            return kite
        except TokenException as e:
            log.warning("ACCESS_TOKEN invalid/expired: %s", e)

    # Fallback to generate_session if secret + request_token supplied
    if api_secret and request_token:
        sess = kite.generate_session(request_token, api_secret)
        new_token = sess.get("access_token")
        if not new_token:
            raise RuntimeError("generate_session returned no access_token")
        kite.set_access_token(new_token)
        try:
            kite.profile()
            log.info("Kite session OK via generate_session; access_token refreshed.")
            return kite
        except TokenException as e:
            raise RuntimeError(f"Token still invalid after generate_session: {e}") from e

    raise RuntimeError(f"Invalid Zerodha credentials. {REQUIRED_MSG}")
