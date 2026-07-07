"""Process- and host-local Telegram polling ownership guard."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import TextIO

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_ACTIVE_TOKEN_HASH: str | None = None
_ACTIVE_OWNER: str | None = None
_LOCK_HANDLE: TextIO | None = None
_LOCK_PATH: Path | None = None


def _token_hash(token: str | None) -> str:
    payload = str(token or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:12]


def _lock_dir() -> Path:
    raw = os.getenv("NIFTY_RUNTIME_LOCK_DIR") or os.getenv("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    path = Path(raw) / "nifty_scalper_bot"
    with suppress(Exception):
        path.mkdir(parents=True, exist_ok=True)
    return path


def _try_host_lock(token_hash: str, owner: str) -> tuple[TextIO | None, Path | None, bool]:
    """Return file handle/path/acquired for a same-host polling lock."""

    try:
        import fcntl  # POSIX only; production host is Linux.
    except Exception:  # pragma: no cover - non-POSIX fallback
        return None, None, True

    path = _lock_dir() / f"telegram-polling-{token_hash}.lock"
    try:
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with suppress(Exception):
                handle.close()
            return None, path, False
        handle.seek(0)
        handle.truncate()
        handle.write(f"owner={owner} pid={os.getpid()} token_hash={token_hash}\n")
        handle.flush()
        return handle, path, True
    except Exception as exc:  # noqa: BLE001 - do not break Telegram startup due to lock FS issues
        _LOG.warning(
            "TELEGRAM_POLLING_HOST_LOCK_UNAVAILABLE path=%s err=%s",
            path,
            exc,
            extra={"event": "TELEGRAM_POLLING_HOST_LOCK_UNAVAILABLE", "token_hash": token_hash},
        )
        return None, path, True


def claim_polling_owner(*, token: str | None, owner: str) -> bool:
    """Return True when *owner* may start polling for *token*.

    The guard is process-local plus same-host lock-file based. It cannot detect a
    different VM/laptop using the same token, but it prevents duplicate services
    inside the same Linux host.
    """

    global _ACTIVE_OWNER, _ACTIVE_TOKEN_HASH, _LOCK_HANDLE, _LOCK_PATH
    token_hash = _token_hash(token)
    with _LOCK:
        if _ACTIVE_TOKEN_HASH == token_hash and _ACTIVE_OWNER not in {None, owner}:
            _LOG.warning(
                "TELEGRAM_DUPLICATE_POLLING_INSTANCE_BLOCKED owner=%s active_owner=%s token_hash=%s",
                owner,
                _ACTIVE_OWNER,
                token_hash,
                extra={"event": "TELEGRAM_DUPLICATE_POLLING_INSTANCE_BLOCKED", "token_hash": token_hash},
            )
            return False
        if _LOCK_HANDLE is None or _ACTIVE_TOKEN_HASH != token_hash:
            handle, path, acquired = _try_host_lock(token_hash, owner)
            if not acquired:
                _LOG.error(
                    "TELEGRAM_DUPLICATE_POLLING_HOST_BLOCKED owner=%s token_hash=%s lock_path=%s",
                    owner,
                    token_hash,
                    path,
                    extra={"event": "TELEGRAM_DUPLICATE_POLLING_HOST_BLOCKED", "token_hash": token_hash},
                )
                return False
            _LOCK_HANDLE = handle
            _LOCK_PATH = path
        _ACTIVE_TOKEN_HASH = token_hash
        _ACTIVE_OWNER = owner
        return True


def release_polling_owner(*, token: str | None, owner: str) -> None:
    """Release polling ownership when the active owner stops."""

    global _ACTIVE_OWNER, _ACTIVE_TOKEN_HASH, _LOCK_HANDLE, _LOCK_PATH
    token_hash = _token_hash(token)
    with _LOCK:
        if _ACTIVE_TOKEN_HASH == token_hash and _ACTIVE_OWNER == owner:
            handle = _LOCK_HANDLE
            if handle is not None:
                with suppress(Exception):
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                with suppress(Exception):
                    handle.close()
            _LOCK_HANDLE = None
            _LOCK_PATH = None
            _ACTIVE_TOKEN_HASH = None
            _ACTIVE_OWNER = None
