"""Fail-closed deployment freshness checks for the trading process.

Railway exposes the source commit as ``RAILWAY_GIT_COMMIT_SHA`` during build and
runtime.  The Docker image embeds that value.  Startup refuses to arm an image
whose embedded commit differs from the deployment commit.  A low-frequency
watchdog also compares the running commit with the configured GitHub branch and
terminates the process after a confirmed mismatch, allowing the platform to
restart or replace the stale instance.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger("nifty_scalper_bot.release_guard")
_UNKNOWN = {"", "unknown", "none", "null", "unset", "not_set"}


def _truthy(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_sha(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in _UNKNOWN:
        return ""
    if all(char in "0123456789abcdef" for char in token) and len(token) >= 7:
        return token
    return ""


def _first_sha(*values: object) -> str:
    for value in values:
        token = normalize_sha(value)
        if token:
            return token
    return ""


def _read_embedded_sha(path: Path | None = None) -> str:
    candidates = [
        path,
        Path(os.getenv("APP_BUILD_SHA_FILE", "/app/.build_commit_sha")),
        Path.cwd() / ".build_commit_sha",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            token = normalize_sha(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if token:
            return token
    return ""


def _same_commit(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


@dataclass(frozen=True, slots=True)
class ReleaseSnapshot:
    build_sha: str
    runtime_sha: str
    effective_sha: str
    repository: str
    branch: str
    strict: bool
    fresh: bool

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["build_sha"] = self.build_sha[:12] if self.build_sha else "unknown"
        payload["runtime_sha"] = self.runtime_sha[:12] if self.runtime_sha else "unknown"
        payload["effective_sha"] = self.effective_sha[:12] if self.effective_sha else "unknown"
        return payload


def build_release_snapshot(*, embedded_path: Path | None = None) -> ReleaseSnapshot:
    build_sha = _first_sha(
        os.getenv("APP_BUILD_SHA"),
        _read_embedded_sha(embedded_path),
    )
    runtime_sha = _first_sha(
        os.getenv("RAILWAY_GIT_COMMIT_SHA"),
        os.getenv("EXPECTED_GIT_COMMIT_SHA"),
        os.getenv("GIT_SHA"),
        os.getenv("RELEASE_ID"),
    )
    railway_runtime = bool(
        os.getenv("RAILWAY_PROJECT_ID")
        or os.getenv("RAILWAY_ENVIRONMENT_NAME")
        or os.getenv("RAILWAY_DEPLOYMENT_ID")
    )
    strict = _truthy(
        os.getenv("RELEASE_GUARD_STRICT"),
        default=railway_runtime,
    )
    repository = (
        os.getenv("RELEASE_GITHUB_REPOSITORY")
        or os.getenv("GITHUB_REPOSITORY")
        or "kundakarlabp/nifty_scalper_bot"
    ).strip()
    branch = (
        os.getenv("RELEASE_WATCH_BRANCH")
        or os.getenv("RAILWAY_GIT_BRANCH")
        or "main"
    ).strip()
    effective_sha = runtime_sha or build_sha
    fresh = bool(build_sha and runtime_sha and _same_commit(build_sha, runtime_sha))
    if not runtime_sha and build_sha:
        fresh = not strict
    if runtime_sha and not build_sha:
        fresh = not strict
    if not runtime_sha and not build_sha:
        fresh = not strict
    return ReleaseSnapshot(
        build_sha=build_sha,
        runtime_sha=runtime_sha,
        effective_sha=effective_sha,
        repository=repository,
        branch=branch,
        strict=strict,
        fresh=fresh,
    )


def enforce_release_freshness(*, embedded_path: Path | None = None) -> ReleaseSnapshot:
    snapshot = build_release_snapshot(embedded_path=embedded_path)
    if snapshot.strict and not snapshot.fresh:
        raise RuntimeError(
            "DEPLOYMENT_RELEASE_MISMATCH "
            f"build_sha={snapshot.build_sha or 'missing'} "
            f"runtime_sha={snapshot.runtime_sha or 'missing'}"
        )
    LOGGER.info(
        "DEPLOYMENT_RELEASE_VERIFIED build_sha=%s runtime_sha=%s strict=%s",
        snapshot.build_sha[:12] or "unknown",
        snapshot.runtime_sha[:12] or "unknown",
        snapshot.strict,
        extra={"event": "DEPLOYMENT_RELEASE_VERIFIED", **snapshot.as_dict()},
    )
    return snapshot


def fetch_remote_branch_sha(
    repository: str,
    branch: str,
    *,
    token: str | None = None,
    timeout: float = 5.0,
) -> str:
    url = f"https://api.github.com/repos/{repository}/commits/{branch}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nifty-scalper-release-watchdog",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed GitHub API host
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        LOGGER.warning(
            "RELEASE_REMOTE_CHECK_FAILED repository=%s branch=%s error=%s",
            repository,
            branch,
            exc,
            extra={"event": "RELEASE_REMOTE_CHECK_FAILED"},
        )
        return ""
    if not isinstance(payload, dict):
        return ""
    return normalize_sha(payload.get("sha"))


def remote_release_is_fresh(
    snapshot: ReleaseSnapshot,
    *,
    fetcher: Callable[[str, str], str] | None = None,
) -> bool | None:
    if not snapshot.effective_sha:
        return None
    if fetcher is None:
        token = os.getenv("GITHUB_RELEASE_WATCH_TOKEN")
        fetcher = lambda repository, branch: fetch_remote_branch_sha(  # noqa: E731
            repository,
            branch,
            token=token,
        )
    remote_sha = fetcher(snapshot.repository, snapshot.branch)
    if not remote_sha:
        return None
    return _same_commit(snapshot.effective_sha, remote_sha)


async def release_watchdog(
    snapshot: ReleaseSnapshot,
    *,
    exit_process: Callable[[int], Any] = os._exit,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> None:
    railway_runtime = bool(os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_ENVIRONMENT_NAME"))
    enabled = _truthy(
        os.getenv("RELEASE_WATCH_ENABLED"),
        default=railway_runtime,
    )
    if not enabled or not snapshot.effective_sha:
        return
    interval = max(60.0, float(os.getenv("RELEASE_WATCH_INTERVAL_SEC", "120") or 120.0))
    initial_delay = max(5.0, float(os.getenv("RELEASE_WATCH_INITIAL_DELAY_SEC", "30") or 30.0))
    await sleep(initial_delay)
    while True:
        fresh = remote_release_is_fresh(snapshot)
        if fresh is False:
            LOGGER.critical(
                "STALE_DEPLOYMENT_DETECTED deployed_sha=%s repository=%s branch=%s exiting=42",
                snapshot.effective_sha[:12],
                snapshot.repository,
                snapshot.branch,
                extra={"event": "STALE_DEPLOYMENT_DETECTED", **snapshot.as_dict()},
            )
            exit_process(42)
            return
        await sleep(interval)


__all__ = [
    "ReleaseSnapshot",
    "build_release_snapshot",
    "enforce_release_freshness",
    "fetch_remote_branch_sha",
    "normalize_sha",
    "release_watchdog",
    "remote_release_is_fresh",
]
