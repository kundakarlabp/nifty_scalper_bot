# Repository-scoped coding skills

These task-specific skills supplement the repository contract for ChatGPT, GitHub Copilot, Codex-compatible tools, and human reviewers.

## Fast start

Read `docs/AGENT_START_HERE.md` and `docs/REPO_MAP.md`. For non-trivial work, generate ranked context with `scripts/agent_context.py` before opening many files.

## Installed skills

| Skill | Purpose | Explicit invocation |
|---|---|---|
| `grill-trading-plan` | Stress-test a fuzzy or safety-critical trading-bot change before PRD/code work | `$grill-trading-plan` |
| `domain-modeling-trading` | Clarify vocabulary, ownership, state, and invariants before design changes | `$domain-modeling-trading` |
| `to-prd-trading-change` | Convert resolved trading-bot context into a concise PRD with non-goals and tests | `$to-prd-trading-change` |
| `to-issues-trading-change` | Split a resolved PRD into small vertical implementation slices | `$to-issues-trading-change` |
| `runtime-contract-validation` | Validate external and cross-module contracts before data/strategy/execution changes | `$runtime-contract-validation` |
| `diagnosing-trading-bugs` | Deterministic diagnosis of runtime, data, signal, broker, and order-state failures | `$diagnosing-trading-bugs` |
| `tdd-trading-changes` | Test-first implementation using one behavior slice at a time | `$tdd-trading-changes` |
| `codebase-design` | Module/interface/seam design while preserving repository ownership | `$codebase-design` |
| `pre-merge-trading-review` | Trading-specific code, backtest, risk, execution, deployment, and merge review | `$pre-merge-trading-review` |
| `session-worklog` | Durable record of decisions, changed files, validation, residual risk, and next action | `$session-worklog` |

Use a skill automatically when the request matches its description, or invoke it explicitly by name in compatible tools.

## Typical sequence

```text
grill-trading-plan
→ domain-modeling-trading
→ to-prd-trading-change
→ to-issues-trading-change
→ diagnosing-trading-bugs when bug-driven
→ tdd-trading-changes
→ codebase-design when an ownership decision is required
→ runtime-contract-validation when touching boundaries/contracts
→ pre-merge-trading-review
→ session-worklog
```

## Relationship to repository instructions

`AGENTS.md` remains authoritative for architecture, trading safety, validation, and merge rules. The skill files provide detailed procedures and should be loaded only when relevant.

## Source and adaptation

The diagnostic, TDD, and codebase-design workflows are adapted from Matt Pocock's `mattpocock/skills` repository and tailored to this NIFTY options scalper. `session-worklog` is vendored from `kundakarlabp/dr-bhanu-prasad` at source commit `c92ac30e6c2e2c7998fd8ebf2669f90b117151a3`.

See `THIRD_PARTY_NOTICE.md` in this directory.
