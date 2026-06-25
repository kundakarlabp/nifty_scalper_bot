# Repo-scoped Codex skills

These skills are discovered automatically by Codex from `.agents/skills/` when Codex is launched anywhere inside this repository.

## Installed skills

| Skill | Purpose | Explicit invocation |
|---|---|---|
| `diagnosing-trading-bugs` | Deterministic diagnosis of runtime, data, signal, broker, and order-state failures | `$diagnosing-trading-bugs` |
| `tdd-trading-changes` | Test-first implementation using one behavior slice at a time | `$tdd-trading-changes` |
| `codebase-design` | Module/interface/seam design while preserving repository ownership | `$codebase-design` |
| `pre-merge-trading-review` | Trading-specific code, backtest, risk, execution, and deployment review | `$pre-merge-trading-review` |

Codex may invoke a skill automatically when the request matches its description. Use `$skill-name` when you need to force a particular workflow.

## Examples

```text
$diagnosing-trading-bugs Diagnose why option ticks are present but readiness remains blocked.
```

```text
$tdd-trading-changes Add idempotent handling for duplicate broker order updates.
```

```text
$codebase-design Review whether reconnect recovery belongs in OrderManager or PositionManager.
```

```text
$pre-merge-trading-review Review this PR for live-trading and backtest validity risks.
```

## Relationship to `AGENTS.md`

`AGENTS.md` remains the repository-wide source of truth. These skills add task-specific execution workflows; they do not override architecture, safety, validation, or merge rules.

## Source and adaptation

The debugging, TDD, and codebase-design workflows are adapted from Matt Pocock's `mattpocock/skills` repository and tailored to this NIFTY options scalper. The source repository is MIT licensed. See `THIRD_PARTY_NOTICE.md` in this directory.