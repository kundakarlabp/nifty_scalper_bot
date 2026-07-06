from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "codebase-design",
    "diagnosing-trading-bugs",
    "domain-modeling-trading",
    "grill-trading-plan",
    "pre-merge-trading-review",
    "runtime-contract-validation",
    "session-worklog",
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
