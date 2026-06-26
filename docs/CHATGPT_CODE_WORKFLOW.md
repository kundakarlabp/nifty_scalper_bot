# ChatGPT GitHub coding workflow

This repository is optimized for code work performed from ordinary ChatGPT chats using the connected GitHub application. A separate personal access token, local clone, Codex workspace, or desktop IDE is not required for the standard workflow.

## Fast access protocol

At the beginning of a task, do not repeatedly test different repository-access methods. Use the GitHub connector as the authoritative read/write path.

Read only these files first:

1. `AGENTS.md`
2. `docs/REPO_MAP.md`
3. `.agents/skills/README.md`

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

These tools only inspect repository structure and produce plans. They do not place orders or start the trading runtime.

## Validation and merge rule

Focused tests provide fast feedback but never replace the full suite. A PR may be squash-merged only when:

- the root cause is documented
- the diff is narrow
- regression coverage exists where practical
- no trading-safety invariant is weakened
- all valid review threads are resolved
- CI passes on the final PR head

When the task explicitly asks ChatGPT to merge after validation, ChatGPT may squash-merge after these conditions are met. Otherwise, leave the PR open for review.

## Access and confidentiality

Do not paste repository credentials or broker session material into chats or issues. The connected GitHub application already supplies repository authorization for supported operations. Repository context reports exclude environment files, runtime data, logs, databases, and key material.
