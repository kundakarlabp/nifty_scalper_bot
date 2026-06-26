# GitHub Copilot instructions — NIFTY Scalper Bot

Read `AGENTS.md`, `docs/REPO_MAP.md`, and `.agents/skills/README.md` before proposing or applying changes.

## Skill routing

- Use `.agents/skills/diagnosing-trading-bugs/SKILL.md` before proposing fixes for runtime, data, readiness, signal, broker, order-state, latency, reconnect, or deployment failures.
- Use `.agents/skills/tdd-trading-changes/SKILL.md` for strategy, market-data, risk, execution, Telegram, and configuration implementation.
- Use `.agents/skills/codebase-design/SKILL.md` for ownership, interfaces, seams, module boundaries, and architecture decisions.
- Use `.agents/skills/pre-merge-trading-review/SKILL.md` for PR, backtest, execution, deployment, and merge review.
- Use `.agents/skills/session-worklog/SKILL.md` after substantial work so later chats recover decisions and validation without replaying the full conversation.

## Hard constraints

- Preserve the authoritative runtime path and module ownership in `AGENTS.md`.
- NIFTY options are the only executable instruments; spot and futures are context only.
- Do not bypass readiness, freshness, quote-quality, risk, cooldown, position, capital, max-loss, or execution-mode gates.
- Do not create duplicate contract selectors, history stores, order paths, state owners, or hidden fallbacks.
- Never add credentials, access tokens, API secrets, broker session material, or real account data.
- Never place a live order during testing.
- Preserve paper, shadow, and live separation.
- Require regression tests, focused validation, the complete repository-required suite, and final-head CI before claiming completion.

When generic suggestions conflict with `AGENTS.md`, repository-local architecture and safety rules take precedence.
