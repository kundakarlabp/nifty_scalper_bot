#!/usr/bin/env python3
# scripts/zerodha_token_cli.py
"""
Automated Zerodha Kite Connect access token helper.

This script automates the retrieval and update of the Zerodha access token.
If the credentials:
  - ZERODHA_USER_ID
  - ZERODHA_PASSWORD
  - ZERODHA_TOTP_SECRET
are set in .env or the system environment, the script will login programmatically
via HTTP requests, generate the TOTP in real-time, retrieve the request_token,
and exchange it for the daily access_token.

If these credentials are not provided, it falls back to manual interactive mode
where it prints the login URL and prompts the user to paste the request_token.

Usage:
  python scripts/zerodha_token_cli.py --write-env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

# Optional load_dotenv helper
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return None

try:
    from kiteconnect import KiteConnect  # type: ignore
except ImportError:
    if __name__ == "__main__":
        sys.exit("Missing dependency: kiteconnect. Install with: pip install kiteconnect")
    KiteConnect = None

try:
    import pyotp
except ImportError:
    if __name__ == "__main__":
        sys.exit("Missing dependency: pyotp. Install with: pip install pyotp")
    pyotp = None

try:
    import requests
except ImportError:
    if __name__ == "__main__":
        sys.exit("Missing dependency: requests. Install with: pip install requests")
    requests = None


def write_env_tokens(env_path: Path, token: str) -> None:
    """Safely updates or adds the access tokens in the .env file."""
    keys_to_update = ["ZERODHA_ACCESS_TOKEN", "BROKER_ACCESS_TOKEN", "KITE_ACCESS_TOKEN"]
    
    if not env_path.exists():
        lines = [f"{key}={token}" for key in keys_to_update]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    content = env_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    
    # Track which keys we've already written so we don't duplicate
    written = set()
    
    for line in lines:
        stripped = line.strip()
        matched_key = None
        for key in keys_to_update:
            if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
                matched_key = key
                break
        
        if matched_key:
            # Overwrite the existing line with the new value
            new_lines.append(f"{matched_key}={token}")
            written.add(matched_key)
        else:
            new_lines.append(line)
            
    # Append any keys that weren't already in the file
    for key in keys_to_update:
        if key not in written:
            new_lines.append(f"{key}={token}")
            
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def get_automated_request_token(
    user_id: str, password: str, totp_secret: str, api_key: str
) -> str | None:
    """Automates the Zerodha web login using a requests session to retrieve the request_token."""
    print("🤖 Launching automated HTTP login flow...")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://kite.zerodha.com/",
    })
    
    # 1. POST Login details
    login_url = "https://kite.zerodha.com/api/login"
    try:
        r1 = session.post(login_url, data={"user_id": user_id, "password": password})
        r1.raise_for_status()
        res1 = r1.json()
    except Exception as exc:
        print(f"❌ Initial login request failed: {exc}", file=sys.stderr)
        return None
        
    if res1.get("status") != "success":
        print(f"❌ Login failed: {res1.get('message', 'Unknown error')}", file=sys.stderr)
        return None
        
    request_id = res1.get("data", {}).get("request_id")
    if not request_id:
        print("❌ Error: No request_id returned from login endpoint.", file=sys.stderr)
        return None
        
    # 2. Generate and POST TOTP
    try:
        # Clean secret of spaces/hyphens
        clean_secret = totp_secret.replace(" ", "").replace("-", "")
        totp = pyotp.TOTP(clean_secret)
        otp = totp.now()
    except Exception as exc:
        print(f"❌ Failed to generate TOTP from secret: {exc}", file=sys.stderr)
        return None
        
    twofa_url = "https://kite.zerodha.com/api/twofa"
    try:
        r2 = session.post(
            twofa_url,
            data={
                "user_id": user_id,
                "request_id": request_id,
                "twofa_value": otp,
                "twofa_type": "totp",
            }
        )
        r2.raise_for_status()
        res2 = r2.json()
    except Exception as exc:
        print(f"❌ Two-factor authentication request failed: {exc}", file=sys.stderr)
        return None
        
    if res2.get("status") != "success":
        print(f"❌ 2FA authentication failed: {res2.get('message', 'Unknown error')}", file=sys.stderr)
        return None
        
    # 3. GET auth/login redirect manually to grab the request_token
    # Using manual redirect tracking avoids ConnectionError on local redirect domains.
    connect_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    current_url = connect_url
    request_token = None
    
    print("🔄 Following login redirects...")
    for _ in range(10):
        try:
            r = session.get(current_url, allow_redirects=False)
            r.raise_for_status()
        except Exception as exc:
            print(f"❌ Redirect request failed: {exc}", file=sys.stderr)
            return None
            
        if r.status_code in (301, 302, 303, 307, 308):
            location = r.headers.get("Location")
            if not location:
                break
            current_url = urljoin(current_url, location)
            
            # Parse parameters to see if we've received the request_token
            parsed = urlparse(current_url)
            qs = parse_qs(parsed.query)
            if "request_token" in qs:
                request_token = qs["request_token"][0]
                break
        else:
            # We hit a terminal page without redirection
            parsed = urlparse(r.url)
            qs = parse_qs(parsed.query)
            if "request_token" in qs:
                request_token = qs["request_token"][0]
            break
            
    return request_token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env", help="Path to .env file (default: ./.env)")
    ap.add_argument(
        "--write-env", action="store_true", default=True, help="Write access tokens to the environment file"
    )
    ap.add_argument("--request-token", help="Provide request_token manually to skip login")
    args = ap.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    # Load API Credentials
    api_key = os.getenv("ZERODHA_API_KEY") or os.getenv("KITE_API_KEY") or os.getenv("BROKER_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET") or os.getenv("KITE_API_SECRET") or os.getenv("BROKER_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Error: ZERODHA_API_KEY and ZERODHA_API_SECRET must be configured in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    request_token = args.request_token

    # Check for automated login credentials if request_token wasn't passed directly
    if not request_token:
        user_id = os.getenv("ZERODHA_USER_ID") or os.getenv("KITE_USER_ID")
        password = os.getenv("ZERODHA_PASSWORD") or os.getenv("KITE_PASSWORD")
        totp_secret = os.getenv("ZERODHA_TOTP_SECRET") or os.getenv("KITE_TOTP_SECRET")
        
        if user_id and password and totp_secret:
            request_token = get_automated_request_token(user_id, password, totp_secret, api_key)
            if not request_token:
                print("⚠️ Automated login failed. Falling back to manual mode...", file=sys.stderr)
        else:
            print("ℹ️ Automated login credentials not complete (needs USER_ID, PASSWORD, and TOTP_SECRET).")

    # Manual fallback
    if not request_token:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
        print("\n=== MANUAL LOGIN REQUIRED ===")
        print("1. Open this URL in your web browser:")
        print(f"   {login_url}")
        print("2. Login and complete the 2FA on the Zerodha page.")
        print("3. You will be redirected to an address (e.g. 127.0.0.1:8000/?status=success&request_token=xyz).")
        try:
            request_token = input("4. Enter the 'request_token' parameter value: ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)

    if not request_token:
        print("❌ Error: No request token supplied.", file=sys.stderr)
        sys.exit(1)

    # 4. Exchange Request Token for Access Token
    print("Exchanging request token for session...")
    try:
        kite = KiteConnect(api_key=api_key)
        session_data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:
        print(f"❌ Failed to exchange request token with Zerodha: {exc}", file=sys.stderr)
        sys.exit(1)

    access_token = session_data.get("access_token")
    if not access_token:
        print("❌ Error: Zerodha session generation did not return an access token.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Successfully authenticated! Access Token: {access_token[:6]}...{access_token[-6:]}")

    if args.write_env:
        try:
            write_env_tokens(env_path, access_token)
            print(f"✅ Successfully wrote access tokens to {env_path.resolve()}")
        except Exception as exc:
            print(f"❌ Failed to write token to {env_path}: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
