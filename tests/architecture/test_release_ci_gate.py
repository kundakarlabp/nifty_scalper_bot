"""Regression tests for the exact-SHA, fail-closed production CI gate."""

from __future__ import annotations

import io
import json
from urllib.error import URLError

from deploy.scripts import verify_release_ci as gate

SHA = "a" * 40
OTHER_SHA = "b" * 40


def _run(
    *,
    sha: str = SHA,
    event: str = "push",
    path: str = ".github/workflows/ci.yml",
    status: str = "completed",
    conclusion: str | None = "success",
    created_at: str = "2026-10-21T13:00:00Z",
) -> dict[str, object]:
    return {
        "head_sha": sha,
        "event": event,
        "path": path,
        "status": status,
        "conclusion": conclusion,
        "created_at": created_at,
        "run_attempt": 1,
    }


def test_only_completed_successful_main_push_ci_authorizes_release() -> None:
    assert gate.classify_ci({"workflow_runs": [_run()]}, SHA) == "passed"
    assert gate.classify_ci({"workflow_runs": [_run(event="pull_request")]}, SHA) == (
        "pending"
    )
    assert gate.classify_ci({"workflow_runs": [_run(sha=OTHER_SHA)]}, SHA) == (
        "pending"
    )
    assert (
        gate.classify_ci(
            {"workflow_runs": [_run(path=".github/workflows/other.yml")]}, SHA
        )
        == "pending"
    )


def test_pending_failed_or_missing_ci_never_authorizes_release() -> None:
    assert gate.classify_ci({"workflow_runs": [_run(status="in_progress")]}, SHA) == (
        "pending"
    )
    assert (
        gate.classify_ci({"workflow_runs": [_run(conclusion="failure")]}, SHA)
        == "failed"
    )
    assert gate.classify_ci({"workflow_runs": []}, SHA) == "pending"
    assert gate.classify_ci({"workflow_runs": "bad"}, SHA) == "unavailable"
    assert gate.classify_ci({}, SHA) == "unavailable"


def test_latest_workflow_result_wins_when_multiple_runs_exist() -> None:
    older = _run(created_at="2026-10-21T12:00:00Z")
    newer = _run(created_at="2026-10-21T14:00:00Z", status="queued")
    assert gate.classify_ci({"workflow_runs": [newer, older]}, SHA) == "pending"


def test_gh_api_unavailable_and_invalid_sha_fail_closed(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr(gate, "urlopen", fail)
    assert gate.verify_ci(SHA) == "unavailable"
    assert gate.verify_ci("not-a-commit") == "unavailable"
    assert gate.main([SHA]) == 1


def test_ci_verifier_queries_exact_main_sha_and_push_run(monkeypatch) -> None:
    called = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_open(request, timeout):
        called.append((request.full_url, timeout))
        return Response(json.dumps({"workflow_runs": [_run()]}).encode())

    monkeypatch.setattr(gate, "urlopen", fake_open)
    assert gate.verify_ci(SHA) == "passed"
    assert gate.main([SHA]) == 0
    url, timeout = called[0]
    assert f"head_sha={SHA}" in url
    assert "event=push" in url
    assert timeout <= 10
