# scripts/zerodha_token_cli.py
"""
Zerodha access token helper.

Steps:
1) Create a KiteConnect app; set redirect URL (any valid).
2) Run this script to print a login URL.
3) Open the URL in a browser, login, authorize; you'll be redirected with ?request_token=...
4) Paste the request_token here; we exchange it for an access_token and (optionally) write to .env.

Usage:
  python scripts/zerodha_token_cli.py --write-env
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from kiteconnect import KiteConnect
except Exception as e:
    raise SystemExit("pip install kiteconnect python-dotenv") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--write-env", action="store_true")
    args = ap.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("ZERODHA_API_KEY") or input("ZERODHA_API_KEY: ").strip()
    api_secret = os.getenv("ZERODHA_API_SECRET") or input("ZERODHA_API_SECRET: ").strip()

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    print("\nOpen this URL and login:\n", login_url, "\n")
    request_token = input("Paste request_token: ").strip()

    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session.get("access_token")
    print("\nACCESS TOKEN:\n", access_token)

    if args.write_env:
        lines = []
        if env_path.exists():
            lines = env_path.read_text().splitlines()
            lines = [ln for ln in lines if not ln.startswith("ZERODHA_ACCESS_TOKEN=")]
        lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}")
        env_path.write_text("\n".join(lines) + "\n")
        print(f"\nWrote ZERODHA_ACCESS_TOKEN to {env_path}")


if __name__ == "__main__":
    main()
