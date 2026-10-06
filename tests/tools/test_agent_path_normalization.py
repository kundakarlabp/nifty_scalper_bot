from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = ROOT / "scripts" / "agent_check.py"


def test_agent_check_normalizes_absolute_source_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    changed_file = (
        repository
        / "src"
        / "nifty_scalper_bot"
        / "streaming"
        / "websocket_manager.py"
    )
    changed_file.parent.mkdir(parents=True)
    changed_file.write_text("class WebSocketManager:\n    pass\n", encoding="utf-8")
    (repository / "tests" / "streaming").mkdir(parents=True)
    (repository / "dashboard").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--root",
            str(repository),
            "--files",
            str(changed_file),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["changed_files"] == [
        "src/nifty_scalper_bot/streaming/websocket_manager.py"
    ]
    assert "streaming" in payload["areas"]
    assert "tests/streaming" in payload["focused_tests"]


def test_agent_check_preserves_leading_dot_directories(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "dashboard").mkdir()
    module_path = repository / ".github" / "workflows" / "ci.yml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("name: CI\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--root",
            str(repository),
            "--files",
            ".github/workflows/ci.yml",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["changed_files"] == [".github/workflows/ci.yml"]
