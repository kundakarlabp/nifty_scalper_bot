"""Process-local Telegram polling ownership guard."""

from __future__ import annotations

import hashlib
import logging
import threading

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_ACTIVE_TOKEN_HASH: str | None = None
_ACTIVE_OWNER: str | None = None


def _token_hash(token: str | None) -> str:
    payload = str(token or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:12]


def claim_polling_owner(*, token: str | None, owner: str) -> bool:
    """Return True when *owner* may start polling for *token*."""

    global _ACTIVE_OWNER, _ACTIVE_TOKEN_HASH
    token_hash = _token_hash(token)
    with _LOCK:
        if _ACTIVE_TOKEN_HASH == token_hash and _ACTIVE_OWNER not in {None, owner}:
            _LOG.warning(
                "TELEGRAM_DUPLICATE_POLLING_INSTANCE_BLOCKED owner=%s active_owner=%s token_hash=%s",
                owner,
                _ACTIVE_OWNER,
                token_hash,
            )
            return False
        _ACTIVE_TOKEN_HASH = token_hash
        _ACTIVE_OWNER = owner
        return True


def release_polling_owner(*, token: str | None, owner: str) -> None:
    """Release polling ownership when the active owner stops."""

    global _ACTIVE_OWNER, _ACTIVE_TOKEN_HASH
    token_hash = _token_hash(token)
    with _LOCK:
        if _ACTIVE_TOKEN_HASH == token_hash and _ACTIVE_OWNER == owner:
            _ACTIVE_TOKEN_HASH = None
            _ACTIVE_OWNER = None
