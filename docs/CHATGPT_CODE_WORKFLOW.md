# ChatGPT GitHub coding workflow

This repository is optimized for code work performed from ordinary ChatGPT chats using the connected GitHub application. A separate personal access token, local clone, Codex workspace, or desktop IDE is not required for the standard workflow.

## Fast access protocol

At the beginning of a task, do not repeatedly test different repository-access methods. Use the GitHub connector as the authoritative read/write path.

Read only these files first:

1. `AGENTS.md`
2. `docs/REPO_MAP.md`
3. `.agents/skills/README.md`

Then load one primary skill from `docs/AGENT_START_HERE.md`. Use a specialist skill only when the task actually matches it; do not load the entire catalog.

For Python/tooling changes, use `docs/ENGINEERING_FAILURE_PATTERNS.md` as the repository's institutional memory. Read only matching pattern IDs. Prefer the automated prevention/detector described there over adding more prompt text.

For a non-trivial error or enhancement, create one GitHub issue with a title beginning:

```text
[Agent Context] exact error, symbol, module, or requested enhancement
```

The `Agent Context Builder` workflow comments on the issue with:

- ranked source files
- matching classes and function signatures
- related regression tests
- risk classification
- suggested focused validation

Use that report to fetch only the highest-ranked files. Avoid broad searches and repeated full-file reads unless the ranked context is insufficient.

## Implementation sequence

```text
GitHub connector available
→ read repository contract and map once
→ obtain compact context report for non-trivial work
→ reproduce the exact symptom
→ establish root cause and ownership
→ create a branch from current main
→ make the smallest architecture-consistent change
→ add regression coverage
→ open one focused PR
→ inspect final diff and automated review
→ require final-head CI
→ squash merge when explicitly requested
```

## Prompt format for efficient tasks

A useful request contains:

```text
Repository: kundakarlabp/nifty_scalper_bot
Observed: exact symptom or sanitized log
Expected: intended behavior
Mode: paper/shadow/live, if relevant
Likely area: optional
Required action: diagnose, implement, validate, open PR, squash merge
Untouched areas: optional
```

Exact error text, event names, class names, function names, Telegram messages, and CI failures are more useful than broad requests such as “optimize everything.”

For recurring specialist work, prefer these repository procedures instead of restating a large prompt:

- `market-data-path-audit` — end-to-end WebSocket FULL/depth/freshness/fallback integrity
- `strategy-research-validation` — strategy logic, evidence, realistic backtesting, and walk-forward validation
- `live-runtime-diagnosis` — read-only production health/status/snapshot/log diagnosis
- `architecture-cleanup` — proven-safe removal of retired or duplicate architecture

## Repository-side commands

Generate a compact context report:

```bash
python scripts/agent_context.py \
  --query "websocket pong timeout reconnect alerts" \
  --output /tmp/agent-context.md
```

Generate a changed-file validation plan:

```bash
python scripts/agent_check.py \
  --files src/nifty_scalper_bot/streaming/websocket_manager.py \
  --output /tmp/agent-check.md
```

Execute the focused ring before publishing Python changes:

```bash
python scripts/agent_check.py \
  --files src/nifty_scalper_bot/streaming/websocket_manager.py \
  --run focused
```

The focused ring is risk-aware: it runs delta-aware Ruff, Black, and mypy checks first, adds architecture ownership validation for production code, and adds deterministic broker-free E2E for high-risk paths. These tools never place orders or start the trading runtime.

## Validation and merge rule

Focused tests provide fast feedback but never replace the full suite. A PR may be squash-merged only when:

- the root cause is documented
- the diff is narrow
- regression coverage exists where practical
- no trading-safety invariant is weakened
- all valid review threads are resolved
- CI passes on the final PR head

When the task explicitly asks ChatGPT to merge after validation, verify the exact validated base/head with `scripts/agent_merge_guard.py` immediately before squash merge. If the base or head moved, refresh and rerun final-head CI. Otherwise, leave the PR open for review.

## Access and confidentiality

Do not paste repository credentials or broker session material into chats or issues. The connected GitHub application already supplies repository authorization for supported operations. Repository context reports exclude environment files, runtime data, logs, databases, and key material.
