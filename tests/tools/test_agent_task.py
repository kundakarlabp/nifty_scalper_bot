from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_task_module():
    path = ROOT / "scripts" / "agent_task.py"
    spec = importlib.util.spec_from_file_location("agent_task_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_agent_task_check_runs_style_validation_and_relevant_benchmarks(
    monkeypatch,
) -> None:
    module = _load_task_module()
    calls: list[tuple[str, tuple[str, ...], bool]] = []

    def fake_python(root, script, *args, capture_output=False):
        calls.append((script, tuple(args), capture_output))
        if script == "agent_check.py" and "--format" in args:
            return _Result(
                stdout=json.dumps(
                    {
                        "areas": ["market-data"],
                        "risk_level": "high",
                    }
                )
            )
        if script == "agent_benchmark.py" and "--format" in args:
            return _Result(
                stdout=json.dumps(
                    {
                        "selected_cases": ["market-data-freshness"],
                        "pytest_targets": ["tests/data"],
                    }
                )
            )
        return _Result()

    monkeypatch.setattr(module, "_python", fake_python)
    monkeypatch.setattr(module, "_syntax_preflight", lambda root: 0)

    result = module._validate(
        ROOT,
        base_ref="origin/main",
        files=("src/nifty_scalper_bot/data/market_data_manager.py",),
        scope="focused",
        style_fix=True,
    )

    assert result == 0
    assert [call[0] for call in calls] == [
        "agent_style_fix.py",
        "agent_check.py",
        "agent_check.py",
        "agent_benchmark.py",
        "agent_benchmark.py",
    ]
    assert "--run" in calls[2][1]
    assert "focused" in calls[2][1]
    assert "--area" in calls[3][1]
    assert "market-data" in calls[3][1]


def test_agent_task_fails_before_style_mutation_when_python_does_not_compile(
    monkeypatch,
) -> None:
    module = _load_task_module()
    calls: list[str] = []

    monkeypatch.setattr(module, "_syntax_preflight", lambda root: 1)

    def fake_python(root, script, *args, capture_output=False):
        calls.append(script)
        return _Result()

    monkeypatch.setattr(module, "_python", fake_python)

    result = module._validate(
        ROOT,
        base_ref="origin/main",
        files=("src/nifty_scalper_bot/strategies/signal_generator.py",),
        scope="focused",
        style_fix=True,
    )

    assert result == 1
    assert calls == []


def test_agent_task_skips_benchmark_execution_when_no_case_matches(monkeypatch) -> None:
    module = _load_task_module()
    calls: list[str] = []

    def fake_python(root, script, *args, capture_output=False):
        calls.append(script)
        if script == "agent_benchmark.py":
            return _Result(
                stdout=json.dumps(
                    {
                        "selected_cases": [],
                        "pytest_targets": [],
                    }
                )
            )
        return _Result()

    monkeypatch.setattr(module, "_python", fake_python)

    assert module._run_benchmarks(ROOT, areas=("agent-tooling",)) == 0
    assert calls == ["agent_benchmark.py"]
