# src/auth/zerodha_auth.py
"""
Zerodha authentication helpers (ASCII-safe, production-friendly).

Provides:
- ZerodhaAuthHandler: tiny HTTP handler to capture `request_token` from the redirect.
- ZerodhaAuthenticator: interactive login flow to obtain and persist an access token.
- get_kite_client(): build a KiteConnect client from env (accepts both token env names).
- check_live_credentials(): quick preflight to see what's missing.

Notes:
- Zerodha access tokens expire DAILY. Generate a fresh one each morning.
- No external dotenv dependency; includes a tiny .env upsert/loader.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

try:
    from kiteconnect import KiteConnect
except Exception as exc:  # pragma: no cover
    raise ImportError("kiteconnect package is required. Install with: pip install kiteconnect") from exc

logger = logging.getLogger(__name__)
if not logger.handlers:
    # If module is run directly, ensure we see logs.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__all__ = [
    "ZerodhaAuthHandler",
    "ZerodhaAuthenticator",
    "check_live_credentials",
    "get_kite_client",
    "load_env_file",
]

# --------------------------- tiny .env helpers --------------------------- #

def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env (no quotes parsing)."""
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k and (k not in os.environ):
                    os.environ[k.strip()] = v.strip()
    except Exception as exc:
        logger.debug("load_env_file(%s) failed: %s", path, exc)


def upsert_env_lines(path: str, kv: dict) -> bool:
    """
    Upsert KEY=VALUE pairs in a .env file, preserving unrelated lines.
    Returns True on success.
    """
    try:
        lines: List[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        # Build a map of first indexes
        idx = {i: line for i, line in enumerate(lines)}
        keys = list(kv.keys())

        # Overwrite in-place if key exists
        found = set()
        for i, line in idx.items():
            if "=" not in line:
                continue
            k = line.split("=", 1)[0].strip()
            if k in kv:
                lines[i] = f"{k}={kv[k]}\n"
                found.add(k)

        # Append missing keys
        for k in keys:
            if k not in found:
                lines.append(f"{k}={kv[k]}\n")

        with open(path, "w", encoding="utf-8") as f:
            # Ensure trailing newline for POSIX-friendliness
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = lines[-1] + "\n"
            f.writelines(lines)
        return True
    except Exception as exc:
        logger.error("Failed to write %s: %s", path, exc, exc_info=True)
        return False


# ------------------------ HTTP callback handler -------------------------- #

class ZerodhaAuthHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive request_token from Zerodha redirect."""

    server_version = "ZerodhaAuthHTTP/1.0"

    def do_GET(self):  # noqa: N802
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        if "request_token" in params:
            request_token = params["request_token"][0]
            setattr(self.server, "request_token", request_token)
            html = (
                "<html><body>"
                "<h2>Authentication Successful</h2>"
                "<p>You can close this window and return to the bot.</p>"
                "</body></html>"
            )
            self._send(200, html)
        else:
            self._send(
                400,
                "<html><body><h2>Authentication Failed</h2>"
                "<p>No request_token found in URL.</p></body></html>",
            )

    def log_message(self, fmt, *args):  # noqa: A003
        try:
            logger.info("HTTP %s - %s", self.address_string(), fmt % args)
        except Exception:
            pass

    def _send(self, status: int, html: str) -> None:
        body = html.encode("utf-8", errors="ignore")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass


# --------------------------- interactive flow ---------------------------- #

def _find_free_port(preferred: int = 8000) -> int:
    """Return preferred if free; otherwise pick an ephemeral port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    except Exception:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class ZerodhaAuthenticator:
    """
    Interactive helper to obtain a Zerodha access token.

    Usage:
        auth = ZerodhaAuthenticator(api_key, api_secret)
        ok = auth.authenticate_interactive()  # opens browser → captures request_token
        if ok:
            auth.save_access_token(".env")     # writes token to env (both names)
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

    def _start_server(self, port: Optional[int] = None) -> Optional[HTTPServer]:
        port = port or _find_free_port(8000)
        try:
            server = HTTPServer(("127.0.0.1", port), ZerodhaAuthHandler)
            setattr(server, "request_token", None)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            logger.info("Local auth server at http://127.0.0.1:%d", port)
            return server
        except Exception as exc:
            logger.error("Failed to start local server: %s", exc, exc_info=True)
            return None

    def authenticate_interactive(self, wait_seconds: int = 150) -> bool:
        """
        Open login URL, wait for request_token via local HTTP.
        Fallback: if server can't be started, prompt for manual paste.
        """
        try:
            server = self._start_server()
            url = self.get_login_url()
            logger.info("Open this URL in your browser to login:\n%s", url)
            try:
                webbrowser.open(url)
            except Exception:
                logger.info("Could not open a browser automatically. Please open the URL manually.")

            if not server:
                # no server → ask user to paste request_token
                return self._manual_request_token_exchange()

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
        except KeyboardInterrupt:
            logger.warning("Authentication interrupted by user.")
            return False
        except Exception as exc:
            logger.error("Interactive authentication failed: %s", exc, exc_info=True)
            return False

    def _manual_request_token_exchange(self) -> bool:
        """Fallback when local server isn't available."""
        try:
            token = input("Paste the 'request_token' from the redirected URL and press Enter: ").strip()
            if not token:
                logger.error("No request_token provided.")
                return False
            self.request_token = token
            return self.generate_access_token()
        except Exception as exc:
            logger.error("Manual exchange failed: %s", exc, exc_info=True)
            return False

    def generate_access_token(self) -> bool:
        """Exchange request_token for access token and set it on the client."""
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
        ok = upsert_env_lines(filepath, {
            "KITE_ACCESS_TOKEN": self.access_token,
            "ZERODHA_ACCESS_TOKEN": self.access_token,
        })
        if ok:
            logger.info("Access token saved to %s (both names).", filepath)
        return ok


# ------------------------------ convenience API --------------------------- #

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
    Construct a KiteConnect client from env:
      ZERODHA_API_KEY, and either ZERODHA_ACCESS_TOKEN or KITE_ACCESS_TOKEN.
    Optionally load a local .env first.
    """
    load_env_file(".env")
    ok, missing = check_live_credentials()
    if not ok:
        raise EnvironmentError("Missing API credentials: " + ", ".join(missing))

    api_key = os.getenv("ZERODHA_API_KEY").strip()
    access_token = (os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
                    or os.getenv("KITE_ACCESS_TOKEN", "").strip())
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# --------------------------------- CLI ------------------------------------ #

def _cli() -> int:
    """
    Handy command line:
      python -m src.auth.zerodha_auth check
      python -m src.auth.zerodha_auth login --env .env
      python -m src.auth.zerodha_auth save --token <ACCESS_TOKEN> --env .env
      python -m src.auth.zerodha_auth client-test
    """
    p = argparse.ArgumentParser(description="Zerodha auth helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Verify required env vars exist")

    p_login = sub.add_parser("login", help="Interactive login flow")
    p_login.add_argument("--env", default=".env", help="Path to .env to update")

    p_save = sub.add_parser("save", help="Save an access token to .env")
    p_save.add_argument("--token", required=True, help="Access token to save")
    p_save.add_argument("--env", default=".env", help="Path to .env to update")

    sub.add_parser("client-test", help="Build a client and fetch profile as a smoke test")

    args = p.parse_args()

    if args.cmd == "check":
        load_env_file(".env")
        ok, missing = check_live_credentials()
        if ok:
            print("✅ Credentials present.")
            return 0
        print("❌ Missing:", ", ".join(missing))
        return 2

    if args.cmd == "login":
        load_env_file(args.env)
        api_key = os.getenv("ZERODHA_API_KEY", "")
        api_secret = os.getenv("ZERODHA_API_SECRET", "")
        if not api_key or not api_secret:
            print("❌ Set ZERODHA_API_KEY and ZERODHA_API_SECRET in env/.env first.")
            return 2
        auth = ZerodhaAuthenticator(api_key, api_secret)
        if not auth.authenticate_interactive():
            return 3
        if not auth.save_access_token(args.env):
            return 4
        print("✅ Access token saved.")
        return 0

    if args.cmd == "save":
        if upsert_env_lines(args.env, {
            "KITE_ACCESS_TOKEN": args.token,
            "ZERODHA_ACCESS_TOKEN": args.token,
        }):
            print(f"✅ Token saved to {args.env}.")
            return 0
        print("❌ Failed to save token.")
        return 4

    if args.cmd == "client-test":
        try:
            kite = get_kite_client()
            prof = kite.profile()  # simple authenticated call
            uid = (prof or {}).get("user_id") or "unknown"
            print(f"✅ Client OK. user_id={uid}")
            return 0
        except Exception as exc:
            print("❌ Client test failed:", exc)
            return 5

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())