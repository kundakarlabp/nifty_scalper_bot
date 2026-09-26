from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


async def test_agents_file_remains_compact_and_routes_to_deeper_docs() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agents.splitlines()) <= 170
    for required in (
        "docs/AGENT_START_HERE.md",
        "docs/REPO_MAP.md",
        "docs/ENGINEERING_FAILURE_PATTERNS.md",
        "docs/CHATGPT_CODE_WORKFLOW.md",
        ".agents/skills/README.md",
    ):
        assert required in agents


async def test_backticked_repository_doc_links_exist() -> None:
    sources = (
        ROOT / "AGENTS.md",
        ROOT / "docs" / "AGENT_START_HERE.md",
        ROOT / "docs" / "CHATGPT_CODE_WORKFLOW.md",
    )
    pattern = re.compile(r"\`((?:docs|\\.agents)/[^\`]+)\`")
    missing: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for value in pattern.findall(text):
            path = value.split("#", 1)[0]
            if "*" in path or "<" in path:
                continue
            if not (ROOT / path).exists():
                missing.append(f"{source.relative_to(ROOT)} -> {path}")
    assert missing == []


async def test_pr_template_requires_validation_and_residual_risk() -> None:
    text = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for heading in (
        "## Objective",
        "## Root cause / rationale",
        "## Invariant",
        "## Explicit non-changes",
        "## Validation",
        "## Failure-memory check",
        "## Residual risk",
    ):
        assert heading in text
