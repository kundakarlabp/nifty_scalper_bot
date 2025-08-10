# src/auth/zerodha_auth.py
"""
Zerodha authentication helpers (ASCII-safe).

Provides:
- ZerodhaAuthHandler: HTTP handler to capture request_token from the redirect.
- ZerodhaAuthenticator: interactive login flow to obtain and persist an access token.
- get_kite_client(): build a KiteConnect client from env (accepts both token env names).
- check_live_credentials(): quick preflight to see what's missing.

Notes:
- Zerodha access tokens expire DAILY. Generate a fresh one each morning.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

try:
    from kiteconnect import KiteConnect
except Exception as exc:
    raise ImportError("kiteconnect package is required. Install with: pip install kiteconnect") from exc

logger = logging.getLogger(__name__)


# --------------------------- HTTP callback handler ---------------------------

class ZerodhaAuthHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive request_token from Zerodha redirect."""

    server_version = "ZerodhaAuthHTTP/1.0"

    def do_GET(self):  # noqa: N802
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if "request_token" in query_params:
            request_token = query_params["request_token"][0]
            setattr(self.server, "request_token", request_token)
            html = (
                "<html><body>"
                "<h2>Authentication Successful</h2>"
                "<p>You can close this window and return to the bot.</p>"
                "</body></html>"
            )
            self._send_html(200, html)
        else:
            html = (
                "<html><body>"
                "<h2>Authentication Failed</h2>"
                "<p>No request_token parameter found in URL.</p>"
                "</body></html>"
            )
            self._send_html(400, html)

    def log_message(self, fmt, *args):  # noqa: A003
        # Use logger instead of printing to stderr
        try:
            logger.info("HTTP %s - %s", self.address_string(), fmt % args)
        except Exception:
            pass

    def _send_html(self, status_code: int, html: str) -> None:
        body = html.encode("utf-8", errors="ignore")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            # ignore broken pipe on browser close
            pass


# --------------------------- Interactive login flow --------------------------

class ZerodhaAuthenticator:
    """
    Interactive helper to obtain a Zerodha access token.

    Usage:
        auth = ZerodhaAuthenticator(api_key, api_secret)
        auth.authenticate_interactive()  # opens browser, captures request_token
        auth.save_access_token(".env")   # writes token to env file (both names)
    """

    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token: Optional[str] = None
        self.request_token: Optional[str] = None

    def get_login_url(self) -> str:
        return self.kite.login_url()

    def start_local_server(self, port: int = 8000) -> Optional[HTTPServer]:
        try:
            server = HTTPServer(("127.0.0.1", port), ZerodhaAuthHandler)
            setattr(server, "request_token", None)
            logger.info("Local auth server at http://127.0.0.1:%d", port)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            return server
        except Exception as exc:
            logger.error("Failed to start local server: %s", exc, exc_info=True)
            return None

    def authenticate_interactive(self, wait_seconds: int = 120) -> bool:
        """Open login URL, capture request_token via local HTTP, exchange for access token."""
        try:
            url = self.get_login_url()
            logger.info("Open this URL in your browser to login: %s", url)
            webbrowser.open(url)

            server = self.start_local_server(8000)
            if not server:
                return False

            start = time.time()
            while getattr(server, "request_token", None) is None:
                if time.time() - start > wait_seconds:
                    logger.error("Timeout waiting for request_token")
                    server.shutdown()
                    return False
                time.sleep(1)

            self.request_token = getattr(server, "request_token")
            logger.info("Received request_token.")
            server.shutdown()

            return self.generate_access_token()
        except Exception as exc:
            logger.error("Interactive authentication failed: %s", exc, exc_info=True)
            return False

    def generate_access_token(self) -> bool:
        """Exchange request_token for access token and set on client."""
        if not self.request_token:
            logger.error("No request_token to exchange.")
            return False
        try:
            data = self.kite.generate_session(self.request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            logger.info("Access token generated and set on Kite client.")
            return True
        except Exception as exc:
            logger.error("Failed to generate access token: %s", exc, exc_info=True)
            return False

    def save_access_token(self, filepath: str = ".env") -> bool:
        """
        Save/update token in a .env-style file under BOTH names:
        - KITE_ACCESS_TOKEN
        - ZERODHA_ACCESS_TOKEN
        """
        if not self.access_token:
            logger.error("No access token to save.")
            return False

        try:
            lines: List[str] = []
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            def upsert(key: str, arr: List[str]) -> List[str]:
                new_line = f"{key}={self.access_token}\n"
                for i, line in enumerate(arr):
                    if line.startswith(key + "="):
                        arr[i] = new_line
                        break
                else:
                    arr.append(new_line)
                return arr

            lines = upsert("KITE_ACCESS_TOKEN", lines)
            lines = upsert("ZERODHA_ACCESS_TOKEN", lines)

            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info("Access token saved to %s (both names).", filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save access token to %s: %s", filepath, exc, exc_info=True)
            return False


# ------------------------------ Convenience API ------------------------------

def check_live_credentials() -> Tuple[bool, List[str]]:
    """
    Returns (ok, missing_list). Accepts either token env var name.
    """
    missing: List[str] = []
    if not os.getenv("ZERODHA_API_KEY"):
        missing.append("ZERODHA_API_KEY")
    if not os.getenv("ZERODHA_API_SECRET"):
        missing.append("ZERODHA_API_SECRET")
    token = os.getenv("ZERODHA_ACCESS_TOKEN") or os.getenv("KITE_ACCESS_TOKEN")
    if not token:
        missing.append("ZERODHA_ACCESS_TOKEN (or KITE_ACCESS_TOKEN)")
    return (len(missing) == 0), missing


def get_kite_client() -> KiteConnect:
    """
    Construct a KiteConnect client from environment variables:
      ZERODHA_API_KEY, ZERODHA_API_SECRET, and either ZERODHA_ACCESS_TOKEN or KITE_ACCESS_TOKEN.

    Raises:
        EnvironmentError if any required variable is missing.
    """
    ok, missing = check_live_credentials()
    if not ok:
        raise EnvironmentError("Missing API credentials: " + ", ".join(missing))

    api_key = os.getenv("ZERODHA_API_KEY").strip()
    access_token = (os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
                    or os.getenv("KITE_ACCESS_TOKEN", "").strip())

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


if __name__ == "__main__":
    # Simple sanity test runner
    has, miss = check_live_credentials()
    if has:
        print("Credentials found. You can build a client with get_kite_client().")
    else:
        print("Missing:", ", ".join(miss))
        print("If you need a new token, instantiate ZerodhaAuthenticator and run authenticate_interactive().")