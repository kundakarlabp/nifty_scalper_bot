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


async def test_current_docs_replace_retired_prompt_and_deployment_duplicates() -> None:
    retired = (
        "docs/codex_prompt.md",
        "docs/mdm_codex_prompts.md",
        "docs/lightsail_validation.md",
        "docs/lightsail_scope2.md",
    )
    assert not [name for name in retired if (ROOT / name).exists()]
    for canonical in (
        "AGENTS.md",
        "docs/AGENT_START_HERE.md",
        "docs/REPO_MAP.md",
        "docs/ENGINEERING_FAILURE_PATTERNS.md",
        "deploy/LIGHTSAIL_SETUP.md",
    ):
        assert (ROOT / canonical).is_file()


async def test_consolidated_incidents_retained_without_stale_fragment_files() -> None:
    retired = (
        "docs/telegram_alert_handler_hygiene.md",
        "docs/telegram_polling_root_cause.md",
        "docs/incident_fixes/websocket_pong_alert_storm.md",
        "docs/market_data_completion_validation.md",
        "docs/future_candle_incident.md",
    )
    assert not [path for path in retired if (ROOT / path).exists()]

    telegram = (ROOT / "docs/telegram_operator_console.md").read_text(encoding="utf-8")
    assert "TELEGRAM_POLLING_CONFLICT" in telegram
    assert "telegram_polling_transient" in telegram
    assert "five-minute monotonic repeat window" in telegram
    assert "root alert handler receives a propagated third-party record" in telegram

    market_data = (ROOT / "docs/market_data_hardening_prd_tdd.md").read_text(
        encoding="utf-8"
    )
    assert "MDM_CANDLE_CLOCK_FLUSHED" in market_data
    assert "CandleStore" in market_data
    assert "out-of-order" in market_data
    assert "test_future_candle_guard.py" in market_data
