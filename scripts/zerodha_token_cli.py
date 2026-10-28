#!/usr/bin/env python3
# scripts/zerodha_token_cli.py
"""Purpose: Maintain a legacy operator-only Zerodha session token helper.

Responsibilities:
- resolve_token_env_path selects the single protected operator credential file.
- write_env_tokens updates the three access-token aliases atomically and privately.
- main exchanges a request token using Zerodha and stores the resulting token.

Boundaries: Not a broker trading, deployment, or service-restart owner.
Safety: Never write an incidental production .env, log token fragments, or arm LIVE.
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys
import tempfile
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


def resolve_token_env_path(
    cli_env: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """Select one operator env file, refusing a different production path."""
    variables = os.environ if environ is None else environ
    protected = variables.get("BOT_ENV_FILE")
    platform = variables.get("DEPLOYMENT_PLATFORM", "")
    production = platform.strip().lower() == "aws_lightsail"
    if production and (not protected or not Path(protected).is_absolute()):
        raise ValueError("canonical BOT_ENV_FILE is required on production")
    candidate = Path(cli_env or protected or ".env").expanduser()
    if production and candidate.resolve() != Path(protected).resolve():
        raise ValueError("only the canonical protected env file is allowed")
    if candidate.is_symlink():
        raise ValueError("symlinked operator env files are not accepted")
    if production and not candidate.is_file():
        raise ValueError("canonical protected env file is missing")
    return candidate


def write_env_tokens(env_path: Path, token: str) -> None:
    """Atomically replace token aliases in one private regular env file."""
    if not token or any(c in token for c in "\r\n"):
        raise ValueError("invalid access token")
    if env_path.is_symlink() or (env_path.exists() and not env_path.is_file()):
        raise ValueError("expected a regular operator env file")
    keys = ("ZERODHA_ACCESS_TOKEN", "BROKER_ACCESS_TOKEN", "KITE_ACCESS_TOKEN")
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    written: set[str] = set()
    updated: list[str] = []
    for line in lines:
        raw = line.strip().removeprefix("export ").strip()
        name = raw.partition("=")[0].strip()
        if name in keys and "=" in raw:
            if name not in written:
                updated.append(f"{name}={token}")
                written.add(name)
        else:
            updated.append(line)
    updated.extend(f"{name}={token}" for name in keys if name not in written)

    # Preserve valid env lines; never truncate the protected file in-place.
    # New inode is private (0600) before it holds any token bytes.
    fd, temp_path = tempfile.mkstemp(prefix=".nifty-token-", dir=env_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write("\n".join(updated) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, env_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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
    ap.add_argument("--env", help="Canonical protected broker environment file")
    ap.add_argument("--write-env", action="store_true", default=True)
    ap.add_argument("--request-token", help="One-time Zerodha request token")
    ap.add_argument(
        "--recover-saved-token",
        action="store_true",
        help="Reverify today's already-saved token and queue the guarded restart",
    )
    args = ap.parse_args()
    if args.recover_saved_token and args.request_token:
        ap.error("choose either --recover-saved-token or --request-token")

    try:
        env_path = resolve_token_env_path(args.env)
    except ValueError as exc:
        sys.exit(f"Token helper configuration refused: {exc}")
    if env_path.is_file():
        # The protected file takes precedence over stale interactive-shell values.
        load_dotenv(env_path, override=True)

    api_key = (
        os.getenv("BROKER_API_KEY")
        or os.getenv("ZERODHA_API_KEY")
        or os.getenv("KITE_API_KEY")
    )
    api_secret = (
        os.getenv("BROKER_API_SECRET")
        or os.getenv("ZERODHA_API_SECRET")
        or os.getenv("KITE_API_SECRET")
    )
    if not api_key or not api_secret:
        sys.exit("Broker API key/secret missing from protected environment")
    key_aliases = ("BROKER_API_KEY", "ZERODHA_API_KEY", "KITE_API_KEY")
    if len({os.getenv(k) for k in key_aliases if os.getenv(k)}) > 1:
        sys.exit("Conflicting broker API-key aliases; refusing session rotation")
    secret_aliases = ("BROKER_API_SECRET", "ZERODHA_API_SECRET", "KITE_API_SECRET")
    if len({os.getenv(k) for k in secret_aliases if os.getenv(k)}) > 1:
        sys.exit("Conflicting broker API-secret aliases; refusing session rotation")

    kite = KiteConnect(api_key=api_key)
    if args.recover_saved_token:
        access_token = (
            os.getenv("BROKER_ACCESS_TOKEN")
            or os.getenv("ZERODHA_ACCESS_TOKEN")
            or os.getenv("KITE_ACCESS_TOKEN")
        )
        if not access_token:
            sys.exit("No saved broker access token available to recover")
    else:
        request_token = args.request_token
        if not request_token:
            user_id = os.getenv("ZERODHA_USER_ID") or os.getenv("KITE_USER_ID")
            password = os.getenv("ZERODHA_PASSWORD") or os.getenv("KITE_PASSWORD")
            totp_secret = os.getenv("ZERODHA_TOTP_SECRET") or os.getenv(
                "KITE_TOTP_SECRET"
            )
            if user_id and password and totp_secret:
                request_token = get_automated_request_token(
                    user_id, password, totp_secret, api_key
                )
            if not request_token:
                print("Open the official Zerodha login URL:", kite.login_url())
                request_token = input("Enter the one-time request token: ").strip()
        if not request_token:
            sys.exit("No request token supplied")
        try:
            access_token = kite.generate_session(
                request_token, api_secret=api_secret
            ).get("access_token")
        except Exception as exc:
            sys.exit(f"Zerodha session exchange failed: {type(exc).__name__}")
        if not access_token:
            sys.exit("Zerodha did not return an access token")

    # A successful token exchange alone is not sufficient. Authenticate the
    # actual runtime API key and owner's broker account before persisting proof.
    kite.set_access_token(access_token)
    try:
        profile = kite.profile()
        verified_user = str(profile.get("user_id") or "").strip()
        if not verified_user:
            raise ValueError("missing broker owner")
        production = (
            os.getenv("DEPLOYMENT_PLATFORM", "").strip().lower() == "aws_lightsail"
        )
        expected_owner = (
            os.getenv("ADMIN_ZERODHA_USER_ID")
            or os.getenv("ZERODHA_USER_ID")
            or os.getenv("KITE_USER_ID")
            or ""
        ).strip()
        owner_mismatch = not expected_owner or not hmac.compare_digest(
            verified_user, expected_owner
        )
        if production and owner_mismatch:
            raise ValueError("owner account mismatch")
        segment = os.getenv("BROKER_MARGIN_SEGMENT", "equity").strip().lower()
        if segment not in {"equity", "commodity"}:
            raise ValueError("invalid broker margin segment")
        margins = kite.margins(segment)
        available = margins.get("available") if isinstance(margins, dict) else None
        if not isinstance(available, dict) or not any(
            available.get(k) is not None for k in ("cash", "live_balance")
        ):
            raise ValueError("broker balance not verifiable")
    except Exception as exc:
        sys.exit(f"Saved session not verified: {type(exc).__name__}")

    print("Zerodha owner and broker funds verified; no credentials displayed.")
    if not args.write_env:
        return
    if production:
        # Reuse the admin's canonical atomic token writer, fresh proof and
        # broker-flat validated release. Never restart systemd directly.
        os.environ["BOT_ENV_FILE"] = str(env_path.resolve())
        from nifty_scalper_bot.admin_dashboard import _activate_access_token

        result = _activate_access_token(access_token)
        print(result.message)
        if not result.ok:
            sys.exit("Token saved but guarded restart could not be queued")
    else:
        write_env_tokens(env_path, access_token)
        print("Access token aliases stored in the selected environment file.")


if __name__ == "__main__":
    main()
