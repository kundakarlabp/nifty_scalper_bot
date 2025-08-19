from __future__ import annotations
import argparse, os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_a, **_k): pass
try:
    from kiteconnect import KiteConnect
except Exception as e:
    raise SystemExit("Missing deps. Install: pip install kiteconnect python-dotenv") from e

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--write-env", action="store_true")
    ap.add_argument("--request-token")
    args = ap.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("ZERODHA_API_KEY") or input("ZERODHA_API_KEY: ").strip()
    api_secret = os.getenv("ZERODHA_API_SECRET") or input("ZERODHA_API_SECRET: ").strip()

    kite = KiteConnect(api_key=api_key)
    print("\nOpen this URL and login:\n", kite.login_url(), "\n")
    req_token = args.request_token or input("Paste request_token: ").strip()

    session = kite.generate_session(req_token, api_secret=api_secret)
    access_token = session.get("access_token")
    print("\nACCESS TOKEN:\n", access_token)

    if args.write_env:
        lines = []
        if env_path.exists():
            lines = [ln for ln in env_path.read_text().splitlines()
                     if not ln.startswith("ZERODHA_ACCESS_TOKEN=")]
        lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote ZERODHA_ACCESS_TOKEN to {env_path}")

if __name__ == "__main__":
    main()
