from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_architecture_module():
    path = ROOT / "scripts" / "agent_architecture.py"
    spec = importlib.util.spec_from_file_location("agent_architecture_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_architecture_manifest_is_valid_and_paths_exist() -> None:
    module = _load_architecture_module()
    payload = module.load_manifest(ROOT)

    assert payload["version"] == 1
    assert len(payload["owners"]) >= 8
    assert len(payload["runtime_path"]) >= 8
    assert all((ROOT / path).exists() for path in payload["runtime_path"])
    assert all((ROOT / item["owner"]).exists() for item in payload["owners"])


def test_agent_tools_do_not_reintroduce_parallel_architecture_tables() -> None:
    agent_check = (ROOT / "scripts" / "agent_check.py").read_text(encoding="utf-8")
    architecture_lint = (ROOT / "scripts" / "architecture_lint.py").read_text(
        encoding="utf-8"
    )

    assert "HIGH_RISK_MARKERS =" not in agent_check
    assert "\nRULES = (" not in agent_check
    assert "BROKER_HISTORY_ALLOWLIST =" not in architecture_lint
    assert "HYDRATION_FORBIDDEN_PREFIXES =" not in architecture_lint
    assert "STRATEGY_FORBIDDEN_CALLS =" not in architecture_lint


def test_human_agent_docs_reference_architecture_ssot() -> None:
    manifest = "docs/architecture/agent_manifest.json"
    for relative in ("AGENTS.md", "docs/REPO_MAP.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert manifest in text
