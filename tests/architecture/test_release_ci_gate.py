"""Regression tests for the exact-SHA, fail-closed production CI gate."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
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
    monkeypatch.setattr(gate, "verified_ci_proof_tag", lambda _sha: False)
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
    monkeypatch.setattr(gate, "verified_ci_proof_tag", lambda _sha: False)
    assert gate.verify_ci(SHA) == "passed"
    assert gate.main([SHA]) == 0
    url, timeout = called[0]
    assert f"head_sha={SHA}" in url
    assert "event=push" in url
    assert timeout <= 10


def test_release_ci_gate_precedes_checkout_and_keeps_exposure_guard() -> None:
    from pathlib import Path

    release = (
        Path(__file__).resolve().parents[2] / "deploy/lightsail_release.sh"
    ).read_text(encoding="utf-8")
    # Even a valid PR/head SHA is not enough: post-merge push CI must pass.
    gate = release.index('verify_release_ci.py" "$AFTER"')
    candidate = release.index('CANDIDATE="/tmp/niftybot-candidate-')
    checkout = release.index('git reset --hard --quiet "$AFTER"')
    assert gate < candidate < checkout
    assert "deferred_ci" in release
    assert "ci_failed" in release
    assert "ci_unavailable" in release
    assert "intraday_flat_broker_confirmed" in release
    assert "auto_release_should_defer" in release
    assert "tests/architecture/test_release_ci_gate.py" in release


def test_git_success_proof_matches_exact_sha_and_skips_rate_limited_api(
    monkeypatch,
) -> None:
    ref = gate.PROOF_TAG_PREFIX + SHA
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command == ["git", "ls-remote", "--refs", "origin", ref]
        assert kwargs["timeout"] <= 10
        return SimpleNamespace(returncode=0, stdout=f"{SHA}\\t{ref}\\n")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "urlopen",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("REST rate")),
    )
    assert gate.verify_ci(SHA) == "passed"
    assert len(calls) == 1


def test_git_proof_cannot_authorize_missing_or_wrong_ref(monkeypatch) -> None:
    outputs = [OTHER_SHA + "\\t" + gate.PROOF_TAG_PREFIX + SHA + "\\n", ""]
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=outputs.pop(0)),
    )
    assert gate.verified_ci_proof_tag(SHA) is False
    assert gate.verified_ci_proof_tag(SHA) is False


def test_workflow_publishes_proof_only_after_complete_main_push_ci() -> None:
    from pathlib import Path

    yml = (
        Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
    assert "publish-main-ci-proof:" in yml
    section = yml.split("  publish-main-ci-proof:", 1)[1].split(
        "  final-validation:", 1
    )[0]
    assert "needs: [final-validation]" in section
    assert "github.event_name == 'push'" in section
    assert "contents: write" in section
    assert 'test "$VALIDATION" = "success"' in section
    assert "refs/tags/nifty-ci-passed/$GITHUB_SHA" in section
    assert "git push origin" in section
    assert "--force" not in section
