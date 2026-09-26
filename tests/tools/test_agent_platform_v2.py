from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def _load(path: str, name: str):
    script = ROOT / path
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_lint_detects_forbidden_strategy_history_call(
    tmp_path: Path,
) -> None:
    module = _load("scripts/architecture_lint.py", "architecture_lint_test")
    root = tmp_path / "repo"
    path = root / "src" / "nifty_scalper_bot" / "strategies" / "bad.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "def run(client):\n    return client.historical_data('x')\n",
        encoding="utf-8",
    )

    violations = module.inspect_file(root, path)

    rules = {item.rule for item in violations}
    assert "broker-history-owner" in rules
    assert "strategy-boundary" in rules


def test_failure_learning_classifies_without_mutating_memory() -> None:
    module = _load("scripts/agent_failure_learn.py", "agent_failure_learn_test")
    counts = module.classify(
        [
            "Ruff changed Python failed E501",
            "Black would reformat test_file.py",
            "mypy error: incompatible types [assignment]",
            "main advanced after CI; stale base",
        ]
    )

    assert counts["LINT-001"] >= 1
    assert counts["FMT-001"] >= 1
    assert counts["TYPE-001"] >= 1
    assert counts["GIT-001"] >= 1


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True)


def _sha(root: Path, ref: str = "HEAD") -> str:
    return subprocess.check_output(
        ["git", "rev-parse", ref],
        cwd=root,
        text=True,
    ).strip()


def _init_repo(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    base = _sha(root)
    _git(root, "branch", "main")
    (root / "a.txt").write_text("head\n", encoding="utf-8")
    _git(root, "commit", "-qam", "head")
    return base, _sha(root)


def test_merge_guard_accepts_exact_validated_base_and_head(tmp_path: Path) -> None:
    module = _load("scripts/agent_merge_guard.py", "agent_merge_guard_test")
    root = tmp_path / "repo"
    base, head = _init_repo(root)

    result = module.verify(
        root,
        validated_base=base,
        validated_head=head,
        base_ref="main",
        head_ref="HEAD",
    )

    assert result["ok"] is True


def test_merge_guard_rejects_stale_base(tmp_path: Path) -> None:
    module = _load("scripts/agent_merge_guard.py", "agent_merge_guard_stale_test")
    root = tmp_path / "repo"
    base, head = _init_repo(root)
    _git(root, "checkout", "-q", "main")
    (root / "b.txt").write_text("advanced\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "advance main")
    _git(root, "checkout", "-q", head)

    result = module.verify(
        root,
        validated_base=base,
        validated_head=head,
        base_ref="main",
        head_ref="HEAD",
    )

    assert result["ok"] is False
    assert result["checks"]["base_matches"] is False
