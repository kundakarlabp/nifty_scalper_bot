# src/auth/zerodha_auth.py

"""
Zerodha authentication helpers.

Provides:
- ZerodhaAuthHandler: HTTP handler to capture the request_token from the redirect.
- ZerodhaAuthenticator: interactive login flow to obtain and persist an access token.
- get_kite_client(): convenience function to create a KiteConnect client from env vars.

Notes:
- All strings are ASCII-safe to avoid "bytes can only contain ASCII literal characters".
- We never place non-ASCII characters inside a b'...' literal.
"""

import os
import logging
import threading
import webbrowser
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    from kiteconnect import KiteConnect
except Exception as exc:
    raise ImportError("kiteconnect package is required. Install with: pip install kiteconnect") from exc

logger = logging.getLogger(__name__)


class ZerodhaAuthHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive request_token from Zerodha redirect."""

    server_version = "ZerodhaAuthHTTP/1.0"

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Store token on the server object so the caller thread can read it
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

    def log_message(self, format, *args):  # noqa: A003 (shadow builtins)
        # Silence BaseHTTPRequestHandler default stdout logging; use Python logger instead
        logger.info("HTTP %s - %s", self.address_string(), format % args)

    def _send_html(self, status_code: int, html: str) -> None:
        try:
            body = html.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            logger.error("Error sending HTTP response: %s", exc, exc_info=True)


class ZerodhaAuthenticator:
    """
    Interactive helper to obtain a Zerodha access token.

    Usage:
        auth = ZerodhaAuthenticator(api_key, api_secret)
        auth.authenticate_interactive()
        auth.save_access_token(".env")
    """

    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token: str | None = None
        self.request_token: str | None = None

    def get_login_url(self) -> str:
        return self.kite.login_url()

    def start_local_server(self, port: int = 8000) -> HTTPServer | None:
        """Start a local HTTP server to capture the redirect with request_token."""
        try:
            server = HTTPServer(("127.0.0.1", port), ZerodhaAuthHandler)
            setattr(server, "request_token", None)
            logger.info("Local auth server started at http://127.0.0.1:%d", port)

            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return server
        except Exception as exc:
            logger.error("Failed to start local server: %s", exc, exc_info=True)
            return None

    def authenticate_interactive(self, wait_seconds: int = 120) -> bool:
        """
        Open the Zerodha login URL in a browser, start a local server to capture the
        request_token, then exchange it for an access token.
        """
        try:
            login_url = self.get_login_url()
            logger.info("Open this URL in your browser to login: %s", login_url)

            webbrowser.open(login_url)
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
            logger.info("Received request_token")
            server.shutdown()

            return self.generate_access_token()
        except Exception as exc:
            logger.error("Interactive authentication failed: %s", exc, exc_info=True)
            return False

    def generate_access_token(self) -> bool:
        """Exchange request_token for an access token and set it on the client."""
        try:
            if not self.request_token:
                logger.error("No request_token to exchange")
                return False

            data = self.kite.generate_session(self.request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)

            logger.info("Access token generated and set on Kite client")
            return True
        except Exception as exc:
            logger.error("Failed to generate access token: %s", exc, exc_info=True)
            return False

    def save_access_token(self, filepath: str = ".env") -> bool:
        """
        Save or update KITE_ACCESS_TOKEN in a .env-style file.
        File writing is ASCII-safe; token content is plain ASCII from API.
        """
        try:
            lines: list[str] = []
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            new_line = f"KITE_ACCESS_TOKEN={self.access_token}\n"
            updated = False

            for i, line in enumerate(lines):
                if line.startswith("KITE_ACCESS_TOKEN="):
                    lines[i] = new_line
                    updated = True
                    break
            if not updated:
                lines.append(new_line)

            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info("Access token saved to %s", filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save access token to %s: %s", filepath, exc, exc_info=True)
            return False


def get_kite_client() -> KiteConnect:
    """
    Construct a KiteConnect client from environment variables:
      ZERODHA_API_KEY, ZERODHA_API_SECRET, KITE_ACCESS_TOKEN

    Raises:
        EnvironmentError if any variable is missing.
    """
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not api_secret or not access_token:
        raise EnvironmentError(
            "Missing API credentials. Set ZERODHA_API_KEY, ZERODHA_API_SECRET, and KITE_ACCESS_TOKEN."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


if __name__ == "__main__":
    # Simple sanity check runner
    API_KEY = os.getenv("ZERODHA_API_KEY")
    API_SECRET = os.getenv("ZERODHA_API_SECRET")

    if API_KEY and API_SECRET:
        auth = ZerodhaAuthenticator(API_KEY, API_SECRET)
        print("Run auth.authenticate_interactive() to start the login flow.")
    else:
        print("Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in environment variables.")