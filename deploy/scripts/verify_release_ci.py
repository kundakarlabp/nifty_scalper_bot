"""Verify that the exact main SHA passed post-merge GitHub CI before release.

This check is read-only, uses no GitHub token, and fails closed if GitHub
cannot establish the state of the authoritative push workflow.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CI_RUNS_URL = (
    "https://api.github.com/repos/kundakarlabp/nifty_scalper_bot/"
    "actions/workflows/ci.yml/runs"
)
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")


def classify_ci(payload: object, sha: str) -> str:
    """Return passed, pending, failed or unavailable for one exact push SHA."""
    if not isinstance(payload, Mapping):
        return "unavailable"
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        return "unavailable"

    eligible = [
        run
        for run in runs
        if isinstance(run, Mapping)
        and run.get("head_sha") == sha
        and run.get("event") == "push"
        and run.get("path") == CI_WORKFLOW_PATH
    ]
    if not eligible:
        return "pending"
    latest = max(
        eligible,
        key=lambda run: (
            str(run.get("created_at") or ""),
            int(run.get("run_attempt") or 0),
        ),
    )
    if latest.get("status") != "completed":
        return "pending"
    if latest.get("conclusion") == "success":
        return "passed"
    return "failed"


def verify_ci(sha: str) -> str:
    """Check public GitHub Actions metadata, never invoke a release."""
    if SHA_PATTERN.fullmatch(sha) is None:
        return "unavailable"
    url = CI_RUNS_URL + "?" + urlencode(
        {"head_sha": sha, "event": "push", "per_page": 10}
    )
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nifty-scalper-release-gate",
        },
    )
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return "unavailable"
    return classify_ci(payload, sha)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    result = verify_ci(args[0]) if len(args) == 1 else "unavailable"
    print(result)
    return 0 if result == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
