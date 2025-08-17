from __future__ import annotations
"""
Zerodha / Kite auth helpers.

What this file does
- Creates a single cached KiteConnect client using env vars.
- Accepts either a ready-made ACCESS TOKEN, or (optionally) exchanges a
  REQUEST_TOKEN + API_SECRET into an access token on boot.
- Trims quotes/whitespace from envs and validates they match.
- Provides a `check_live_credentials()` preflight used by CLI `auth-check`.

Relevant env variables (any not needed can be left blank):
- ZERODHA_API_KEY            (required for any live call)
- KITE_ACCESS_TOKEN          (preferred ready access token)
- ZERODHA_ACCESS_TOKEN       (legacy name; used if KITE_ACCESS_TOKEN empty)
- ZERODHA_API_SECRET         (optional; for exchanging REQUEST_TOKEN)
- ZERODHA_REQUEST_TOKEN      (optional; if present + secret -> exchange)
"""

import os
import threading
from typing import Optional, Tuple

from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, InputException

_KITE_LOCK = threading.RLock()
_KITE_CLIENT: Optional[KiteConnect] = None
_LAST_ACCESS_TOKEN: Optional[str] = None
_LAST_API_KEY: Optional[str] = None


def _clean(x: Optional[str]) -> str:
    if not x:
        return ""
    return x.strip().strip('"').strip("'").strip()


def _mask(x: str, keep: int = 4) -> str:
    x = x or ""
    if len(x) <= keep:
        return "*" * len(x)
    return "*" * (len(x) - keep) + x[-keep:]


def _read_creds() -> Tuple[str, str, str, str]:
    api_key = _clean(os.getenv("ZERODHA_API_KEY"))
    access_token = _clean(os.getenv("KITE_ACCESS_TOKEN")) or _clean(os.getenv("ZERODHA_ACCESS_TOKEN"))
    api_secret = _clean(os.getenv("ZERODHA_API_SECRET"))
    request_token = _clean(os.getenv("ZERODHA_REQUEST_TOKEN"))
    return api_key, access_token, api_secret, request_token


def _exchange_request_token(api_key: str, api_secret: str, request_token: str) -> str:
    """
    Exchange REQUEST_TOKEN + API_SECRET into an ACCESS TOKEN (valid for the day).
    NOTE: REQUEST_TOKEN is short-lived and obtained after user login to the Kite
    redirect URL. This helper simply finishes the exchange.
    """
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token=request_token, api_secret=api_secret)
    return str(data["access_token"])


def get_kite_client(force_refresh: bool = False) -> KiteConnect:
    """
    Return a cached, authenticated KiteConnect client.
    Prefers ready ACCESS TOKEN. If not present but REQUEST_TOKEN + SECRET are
    available, exchanges them automatically.

    Raises a meaningful exception if something is wrong.
    """
    global _KITE_CLIENT, _LAST_ACCESS_TOKEN, _LAST_API_KEY

    with _KITE_LOCK:
        api_key, access_token, api_secret, request_token = _read_creds()

        if not api_key:
            raise RuntimeError("ZERODHA_API_KEY is missing")

        # If cached, and creds didn't change, return it
        if (
            _KITE_CLIENT is not None
            and not force_refresh
            and _LAST_API_KEY == api_key
            and _LAST_ACCESS_TOKEN == access_token
        ):
            return _KITE_CLIENT

        # If no access token but we have a request_token + secret, try exchange.
        if not access_token and api_secret and request_token:
            try:
                access_token = _exchange_request_token(api_key, api_secret, request_token)
                # Store it back into env so other parts can use it too (optional)
                os.environ["KITE_ACCESS_TOKEN"] = access_token
            except Exception as e:
                raise RuntimeError(f"Failed to exchange REQUEST_TOKEN: {e}")

        if not access_token:
            raise RuntimeError(
                "Kite ACCESS TOKEN not found. Set KITE_ACCESS_TOKEN (or ZERODHA_ACCESS_TOKEN), "
                "or provide ZERODHA_REQUEST_TOKEN + ZERODHA_API_SECRET to exchange."
            )

        # Build fresh client
        kite = KiteConnect(api_key=api_key)
        try:
            kite.set_access_token(access_token)
            # Light sanity call – profile is cheap/authoritative; margins also ok
            kite.profile()
        except TokenException as te:
            # Most common cause: expired or mismatched token (different api_key)
            raise RuntimeError(
                "Kite access token rejected (likely expired or does not match API key). "
                f"api_key={_mask(api_key)}, access_token={_mask(access_token)} | {te}"
            )
        except InputException as ie:
            # Misconfig or validation error
            raise RuntimeError(
                f"Kite input error. api_key={_mask(api_key)}, access_token={_mask(access_token)} | {ie}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Kite client init failed. api_key={_mask(api_key)}, access_token={_mask(access_token)} | {e}"
            )

        _KITE_CLIENT = kite
        _LAST_API_KEY = api_key
        _LAST_ACCESS_TOKEN = access_token
        return kite


def check_live_credentials() -> Tuple[bool, list[str]]:
    """
    Quick preflight used by CLI `auth-check`.
    Returns (ok, missing_list). If ok is False and nothing is missing, it means
    something is invalid (like an expired token) and the exception message will help.
    """
    missing: list[str] = []
    api_key, access_token, api_secret, request_token = _read_creds()
    if not api_key:
        missing.append("ZERODHA_API_KEY")
    if not access_token and not (api_secret and request_token):
        # Need either an access token OR (request_token + secret) to proceed
        missing.append("KITE_ACCESS_TOKEN or (ZERODHA_REQUEST_TOKEN + ZERODHA_API_SECRET)")

    if missing:
        return False, missing

    # Try building the client (will raise on mismatch/expiry)
    try:
        get_kite_client(force_refresh=True)
        return True, []
    except Exception as e:
        # Surface the exact problem to logs (caller prints)
        raise