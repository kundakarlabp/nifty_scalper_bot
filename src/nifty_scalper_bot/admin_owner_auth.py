"""Purpose: Authorize one operator browser through verified Zerodha ownership.

Responsibilities:
- Issue signed short-lived login challenges and owner browser sessions.
- Persist a random signing key in the host's private configuration directory.
- Never treat the Kite request token, public API key, or a guessed user ID as authority.

Boundaries: No broker orders, deployment or credential mutation in this module.
Safety: HTTPS, owner allowlist, login challenge and HttpOnly cookies are mandatory.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response

LOGIN_COOKIE = "nifty_owner_login"
SESSION_COOKIE = "nifty_owner_session"
LOGIN_TTL_SECONDS = 300
SESSION_TTL_SECONDS = 36000


def _signing_key() -> bytes:
    """Load or safely create the independent 256-bit server secret."""
    directory = Path(
        os.getenv("NIFTYBOT_CONFIG_DIR", "/home/ubuntu/.config/niftybot")
    ).expanduser()
    path = directory / "owner_session_key"
    if not directory.is_dir() or directory.is_symlink():
        raise HTTPException(status_code=503, detail="owner_session_config_unavailable")
    try:
        if not path.exists() and not path.is_symlink():
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(secrets.token_bytes(32))
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise OSError("session key must be private and regular")
        key = path.read_bytes()
        if len(key) != 32:
            raise OSError("invalid session key size")
        return key
    except OSError:
        raise HTTPException(
            status_code=503, detail="owner_session_config_unavailable"
        ) from None


def _pack(payload: dict[str, object]) -> str:
    value = (
        base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        .decode()
        .rstrip("=")
    )
    mac = hmac.new(_signing_key(), value.encode(), hashlib.sha256).hexdigest()
    return value + "." + mac


def _unpack(value: str) -> dict[str, object] | None:
    if not value or len(value) > 1024:
        return None
    try:
        body, signature = value.split(".", 1)
        expected = hmac.new(_signing_key(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if not isinstance(payload, dict):
            return None
        issued = payload.get("iat")
        if not isinstance(issued, int) or issued > time.time() + 10:
            return None
        return payload
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


def require_https(request: Request) -> None:
    # FastAPI trusts proxy headers only from the configured loopback Caddy hop.
    if request.url.scheme != "https":
        raise HTTPException(status_code=403, detail="owner_https_required")


def owner_user_id(env: dict[str, str]) -> str:
    """Fail closed until the account has been explicitly owner-allowlisted."""
    return str(
        env.get("ADMIN_ZERODHA_USER_ID")
        or env.get("ZERODHA_USER_ID")
        or env.get("KITE_USER_ID")
        or os.getenv("ADMIN_ZERODHA_USER_ID", "")
    ).strip()


def create_challenge(response: Response) -> str:
    token = _pack(
        {"kind": "login", "nonce": secrets.token_urlsafe(24), "iat": int(time.time())}
    )
    response.set_cookie(
        LOGIN_COOKIE,
        token,
        secure=True,
        httponly=True,
        samesite="lax",
        max_age=LOGIN_TTL_SECONDS,
        path="/admin",
    )
    return token


def verify_challenge(request: Request, submitted: str) -> bool:
    cookie = request.cookies.get(LOGIN_COOKIE, "")
    if not cookie or not hmac.compare_digest(cookie, submitted):
        return False
    payload = _unpack(cookie)
    issued_at = payload.get("iat") if payload else None
    return bool(
        payload
        and isinstance(issued_at, int)
        and payload.get("kind") == "login"
        and isinstance(payload.get("nonce"), str)
        and time.time() - issued_at <= LOGIN_TTL_SECONDS
    )



def require_challenge_bound_form(request: Request, submitted: str) -> None:
    """Protect a mobile token POST even when referrer headers are suppressed.

    A signed challenge must match a Secure, SameSite browser cookie. Treat
    Origin/Referer as extra checks when supplied, not mandatory credentials.
    Explicit foreign origins remain forbidden, including cross-site Referer.
    """
    require_https(request)
    if not verify_challenge(request, submitted):
        raise HTTPException(status_code=403, detail="owner_login_challenge_invalid")

    expected_host = request.url.hostname
    expected_port = request.url.port or 443
    for header in ("origin", "referer"):
        source = request.headers.get(header, "").strip()
        if not source or source == "null":
            # Browser privacy policies may omit either header; authentication
            # still requires the cryptographically signed form+cookie pair.
            continue
        try:
            parsed = urlsplit(source)
            valid = (
                parsed.scheme == "https"
                and parsed.hostname == expected_host
                and (parsed.port or 443) == expected_port
                and not parsed.username
                and not parsed.password
            )
        except ValueError:
            valid = False
        if not valid:
            raise HTTPException(status_code=403, detail="admin_same_origin_required")


def clear_challenge(response: Response) -> None:
    response.delete_cookie(
        LOGIN_COOKIE, path="/admin", secure=True, httponly=True, samesite="lax"
    )


def grant_owner_session(response: Response, user_id: str) -> None:
    cookie = _pack({"kind": "session", "uid": user_id, "iat": int(time.time())})
    response.set_cookie(
        SESSION_COOKIE,
        cookie,
        secure=True,
        httponly=True,
        samesite="strict",
        max_age=SESSION_TTL_SECONDS,
        path="/admin",
    )


def valid_owner_session(request: Request, expected_user_id: str) -> bool:
    if not expected_user_id or request.url.scheme != "https":
        return False
    payload = _unpack(request.cookies.get(SESSION_COOKIE, ""))
    issued_at = payload.get("iat") if payload else None
    return bool(
        payload
        and isinstance(issued_at, int)
        and payload.get("kind") == "session"
        and isinstance(payload.get("uid"), str)
        and hmac.compare_digest(str(payload["uid"]), expected_user_id)
        and time.time() - issued_at <= SESSION_TTL_SECONDS
    )
