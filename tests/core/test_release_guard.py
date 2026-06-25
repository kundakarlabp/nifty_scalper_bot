from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nifty_scalper_bot.core.release_guard import (
    ReleaseSnapshot,
    build_release_snapshot,
    enforce_release_freshness,
    normalize_sha,
    remote_release_is_fresh,
    start_release_watchdog_thread,
)


SHA_A = "a" * 40
SHA_B = "b" * 40


def _clear_release_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "APP_BUILD_SHA",
        "APP_BUILD_SHA_FILE",
        "RAILWAY_GIT_COMMIT_SHA",
        "EXPECTED_GIT_COMMIT_SHA",
        "GIT_SHA",
        "RELEASE_ID",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_DEPLOYMENT_ID",
        "RELEASE_GUARD_STRICT",
        "RELEASE_WATCH_ENABLED",
        "RELEASE_WATCH_INTERVAL_SEC",
        "RELEASE_WATCH_INITIAL_DELAY_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


def test_strict_release_guard_accepts_matching_build_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_release_env(monkeypatch)
    marker = tmp_path / ".build_commit_sha"
    marker.write_text(SHA_A, encoding="utf-8")
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "project")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", SHA_A)

    snapshot = enforce_release_freshness(embedded_path=marker)

    assert snapshot.strict is True
    assert snapshot.fresh is True
    assert snapshot.effective_sha == SHA_A


def test_strict_release_guard_rejects_mismatched_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_release_env(monkeypatch)
    marker = tmp_path / ".build_commit_sha"
    marker.write_text(SHA_A, encoding="utf-8")
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "project")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", SHA_B)

    with pytest.raises(RuntimeError, match="DEPLOYMENT_RELEASE_MISMATCH"):
        enforce_release_freshness(embedded_path=marker)


def test_local_runtime_without_release_metadata_is_non_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_release_env(monkeypatch)
    snapshot = build_release_snapshot(embedded_path=tmp_path / "missing")
    assert snapshot.strict is False
    assert snapshot.fresh is True
    assert snapshot.effective_sha == ""


def test_remote_release_comparison_is_tristate() -> None:
    snapshot = ReleaseSnapshot(
        build_sha=SHA_A,
        runtime_sha=SHA_A,
        effective_sha=SHA_A,
        repository="kundakarlabp/nifty_scalper_bot",
        branch="main",
        strict=True,
        fresh=True,
    )
    assert remote_release_is_fresh(snapshot, fetcher=lambda _repo, _branch: SHA_A) is True
    assert remote_release_is_fresh(snapshot, fetcher=lambda _repo, _branch: SHA_B) is False
    assert remote_release_is_fresh(snapshot, fetcher=lambda _repo, _branch: "") is None


def test_watchdog_exits_only_after_confirmed_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_release_env(monkeypatch)
    monkeypatch.setenv("RELEASE_WATCH_ENABLED", "true")
    snapshot = ReleaseSnapshot(
        build_sha=SHA_A,
        runtime_sha=SHA_A,
        effective_sha=SHA_A,
        repository="kundakarlabp/nifty_scalper_bot",
        branch="main",
        strict=True,
        fresh=True,
    )
    exits: list[int] = []
    sleeps: list[float] = []
    thread = start_release_watchdog_thread(
        snapshot,
        exit_process=exits.append,
        fetcher=lambda _repo, _branch: SHA_B,
        sleep=lambda seconds: sleeps.append(seconds),
    )
    assert thread is not None
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert exits == [42]
    assert sleeps


def test_normalize_sha_rejects_non_commit_tokens() -> None:
    assert normalize_sha(SHA_A.upper()) == SHA_A
    assert normalize_sha("unknown") == ""
    assert normalize_sha("not-a-sha") == ""
