# GitHub Copilot instructions — NIFTY Scalper Bot

Start with `docs/AGENT_START_HERE.md`, `docs/REPO_MAP.md`, and `.agents/skills/README.md`. Read the full `AGENTS.md` before changing high-risk runtime paths.

## Fast context

- Do not repeat repository discovery after access is confirmed.
- Prefer exact symbols, event names, and log messages.
- Use `scripts/agent_context.py` to rank files, signatures, and related tests.
- Use `scripts/agent_check.py` to map changed files to focused validation.
- Fetch only the ranked files and their direct call sites before broad searches.

## Skill routing

- Runtime/data/readiness/order failures: `.agents/skills/diagnosing-trading-bugs/SKILL.md`
- Test-first implementation: `.agents/skills/tdd-trading-changes/SKILL.md`
- Ownership and interfaces: `.agents/skills/codebase-design/SKILL.md`
- PR and merge review: `.agents/skills/pre-merge-trading-review/SKILL.md`
- Durable continuity: `.agents/skills/session-worklog/SKILL.md`

## Hard constraints

- Preserve the authoritative runtime path and module ownership.
- NIFTY options are the only executable instruments; spot and futures are context only.
- Do not bypass freshness, quote quality, readiness, risk, cooldown, position, capital, max-loss, or execution-mode gates.
- Do not create duplicate selectors, history stores, order paths, state owners, or hidden fallbacks.
- Never place a live order during tests.
- Preserve paper, shadow, and live separation.
- Require regression coverage, focused checks, the complete suite, and final-head CI before completion.

Repository-local rules take precedence over generic coding suggestions.
