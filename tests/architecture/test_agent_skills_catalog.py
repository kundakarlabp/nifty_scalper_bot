from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "architecture-cleanup",
    "codebase-design",
    "diagnosing-trading-bugs",
    "domain-modeling-trading",
    "grill-trading-plan",
    "live-runtime-diagnosis",
    "market-data-path-audit",
    "pre-merge-trading-review",
    "runtime-contract-validation",
    "session-worklog",
    "strategy-research-validation",
    "tdd-trading-changes",
    "to-issues-trading-change",
    "to-prd-trading-change",
}


def _parse_frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "missing opening YAML delimiter"
    raw, _body = text[4:].split("\n---\n", 1)
    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Malformed frontmatter line (missing colon): {line}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def test_expected_agent_skill_catalog_is_present() -> None:
    discovered = {
        path.name
        for path in SKILLS_ROOT.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    assert discovered == EXPECTED_SKILLS


def test_agent_skills_have_activation_metadata_and_core_sections() -> None:
    for name in sorted(EXPECTED_SKILLS):
        skill_file = SKILLS_ROOT / name / "SKILL.md"
        text = skill_file.read_text(encoding="utf-8")
        metadata = _parse_frontmatter(text)
        assert metadata.get("name") == name
        assert len(metadata.get("description", "")) >= 40
        assert re.search(r"^## .+", text, re.MULTILINE)
        assert text.endswith("\n")


def test_skill_readme_routes_all_installed_skills() -> None:
    readme = (SKILLS_ROOT / "README.md").read_text(encoding="utf-8")
    for name in EXPECTED_SKILLS:
        assert f"`{name}`" in readme
    assert "kundakarlabp/dr-bhanu-prasad" in readme


def test_specialist_skills_are_routed_from_agent_start_here() -> None:
    router = (ROOT / "docs" / "AGENT_START_HERE.md").read_text(encoding="utf-8")
    for name in {
        "architecture-cleanup",
        "live-runtime-diagnosis",
        "market-data-path-audit",
        "strategy-research-validation",
    }:
        assert f"`{name}`" in router

def test_engineering_failure_memory_is_wired_into_agent_workflows() -> None:
    memory_path = ROOT / "docs" / "ENGINEERING_FAILURE_PATTERNS.md"
    memory = memory_path.read_text(encoding="utf-8")
    for pattern_id in {
        "FMT-001",
        "LINT-001",
        "TYPE-001",
        "SYNTAX-001",
        "SCOPE-001",
        "TEST-001",
        "TEST-002",
        "GIT-001",
        "DATA-001",
        "STRAT-001",
        "ARCH-001",
        "ORDER-001",
        "OBS-001",
    }:
        assert pattern_id in memory

    for relative_path in (
        "AGENTS.md",
        "docs/AGENT_START_HERE.md",
        "docs/CHATGPT_CODE_WORKFLOW.md",
        ".agents/skills/tdd-trading-changes/SKILL.md",
        ".agents/skills/pre-merge-trading-review/SKILL.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "docs/ENGINEERING_FAILURE_PATTERNS.md" in text

