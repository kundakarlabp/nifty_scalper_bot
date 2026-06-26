from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_SCRIPT = ROOT / "scripts" / "agent_context.py"
CHECK_SCRIPT = ROOT / "scripts" / "agent_check.py"


def _sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    source = root / "src" / "nifty_scalper_bot" / "streaming"
    tests = root / "tests" / "streaming"
    source.mkdir(parents=True)
    tests.mkdir(parents=True)
    (root / ".env").write_text("RUNTIME_VALUE=private-fixture\n", encoding="utf-8")
    (source / "websocket_manager.py").write_text(
        """from __future__ import annotations

class WebSocketManager:
    def watchdog_pong_timeout(self, age: float) -> bool:
        return age > 10
""",
        encoding="utf-8",
    )
    (tests / "test_websocket_manager.py").write_text(
        """from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager

def test_watchdog_pong_timeout():
    assert WebSocketManager().watchdog_pong_timeout(11)
""",
        encoding="utf-8",
    )
    (root / "dashboard").mkdir()
    return root


def test_agent_context_ranks_symbols_and_tests_without_runtime_values(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    report = tmp_path / "context.md"
    subprocess.run(
        [
            sys.executable,
            str(CONTEXT_SCRIPT),
            "--root",
            str(root),
            "--query",
            "websocket_pong_timeout watchdog age",
            "--output",
            str(report),
        ],
        check=True,
    )
    text = report.read_text(encoding="utf-8")
    assert "src/nifty_scalper_bot/streaming/websocket_manager.py" in text
    assert "WebSocketManager.watchdog_pong_timeout" in text
    assert "tests/streaming/test_websocket_manager.py" in text
    assert "private-fixture" not in text
    assert "| `.env` |" not in text


def test_agent_context_json_is_machine_readable(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(CONTEXT_SCRIPT),
            "--root",
            str(root),
            "--query",
            "websocket timeout",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ranked_files"][0]["path"].endswith("websocket_manager.py")
    assert payload["commands"][-1] == "python -m pytest -q  # mandatory before merge"


def test_agent_check_builds_focused_plan(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    (root / "tests" / "data").mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--root",
            str(root),
            "--files",
            "src/nifty_scalper_bot/streaming/websocket_manager.py",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "streaming" in payload["areas"]
    assert "tests/streaming" in payload["focused_tests"]
    assert payload["full_suite_required"] is True
    assert payload["commands"][-1] == "python -m pytest -q"
