from __future__ import annotations

import importlib.util
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


def _load_check_module():
    spec = importlib.util.spec_from_file_location("agent_check_test_module", CHECK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_agent_check_routes_agent_docs_to_tooling_tests(tmp_path: Path) -> None:
    root = _sample_repo(tmp_path)
    (root / "tests" / "tools").mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--root",
            str(root),
            "--files",
            "docs/AGENT_START_HERE.md",
            "docs/AI_OPTIMIZATION_WORKFLOW.md",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "agent-tooling" in payload["areas"]
    assert "tests/tools" in payload["focused_tests"]


def test_agent_check_run_scope_executes_only_requested_ring(
    tmp_path: Path, monkeypatch
) -> None:
    root = _sample_repo(tmp_path)
    module = _load_check_module()
    plan = module.Plan(
        changed_files=("scripts/agent_check.py",),
        areas=("agent-tooling",),
        focused_tests=("tests/tools",),
        commands=(
            "python -m compileall -q src dashboard",
            "python -m pytest -q tests/tools",
            "python -m pytest -q",
        ),
    )
    calls: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return Result()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.run_plan(root, plan, "focused") == 0
    assert len(calls) == 2
    assert calls[0][0] == sys.executable

    calls.clear()
    assert module.run_plan(root, plan, "full") == 0
    assert len(calls) == 3
