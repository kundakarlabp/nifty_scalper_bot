# scripts/zerodha_token_cli.py
"""
Zerodha access token helper.

Steps:
1) Create a KiteConnect app (note API key/secret); set a valid redirect URL.
2) Run this to print a login URL.
3) Open the URL, login, authorize; you'll be redirected with ?request_token=...
4) Provide that request_token (prompt or --request-token) to get an access_token.

Usage:
  python -m src.scripts.zerodha_token_cli --write-env
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from getpass import getpass
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*_a, **_k):  # no-op if dotenv not installed
        return None

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception as e:
    sys.exit("Missing deps. Install with: pip install kiteconnect python-dotenv")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env", help="Path to .env file (default: ./.env)")
    ap.add_argument("--write-env", action="store_true", help="Write KITE_ACCESS_TOKEN into .env")
    ap.add_argument("--request-token", help="Provide request_token non-interactively")
    ap.add_argument("--open", action="store_true", help="Open login URL in a browser")
    args = ap.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("KITE_API_KEY") or input("KITE_API_KEY: ").strip()
    api_secret = os.getenv("KITE_API_SECRET") or getpass("KITE_API_SECRET (hidden): ").strip()

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    print("\nOpen this URL and login:\n", login_url, "\n")
    if args.open:
        try:
            webbrowser.open(login_url)
        except Exception:
            pass

    request_token = args.request_token or input("Paste request_token: ").strip()
    if not request_token:
        sys.exit("No request_token supplied.")

    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        sys.exit(f"Failed to exchange request_token: {e}")

    access_token = session.get("access_token")
    if not access_token:
        sys.exit("No access_token returned by Kite.")

    print("\nACCESS TOKEN:\n", access_token)

    if args.write_env:
        lines = []
        if env_path.exists():
            lines = env_path.read_text().splitlines()
            lines = [ln for ln in lines if not ln.startswith("KITE_ACCESS_TOKEN=")]
        lines.append(f"KITE_ACCESS_TOKEN={access_token}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote KITE_ACCESS_TOKEN to {env_path}")

if __name__ == "__main__":
    main()
