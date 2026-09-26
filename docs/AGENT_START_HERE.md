# Agent start here

Use the smallest amount of repository context that can safely answer the task.

## Fast path

1. Read `docs/REPO_MAP.md`.
2. Rank relevant files and tests before broad browsing:

   ```bash
   python scripts/agent_context.py --query "exact error, symbol, class, function, or behavior"
   ```

3. Load **one primary skill** for the task. Add a secondary skill only when the task actually crosses that concern.
4. For Python/tooling edits, scan the matching entries in `docs/ENGINEERING_FAILURE_PATTERNS.md`; do not load unrelated patterns.
5. Read full `AGENTS.md` before editing a high-risk runtime path.
6. Validate the changed surface first. `agent_check.py` classifies risk, runs delta-aware Ruff/Black/mypy, and adds architecture/E2E checks when warranted.
7. Require full validation and final-head CI before merge, then run `agent_merge_guard.py` against the exact validated base/head.

## Minimal skill routing

| Task | Primary skill | Add only when needed |
|---|---|---|
| Runtime bug, failed test, stale data, wrong signal, duplicate order, unexplained no-trade state | `diagnosing-trading-bugs` | `runtime-contract-validation` for boundary payloads; `codebase-design` for ownership/seam defects |
| WebSocket FULL/depth/freshness/fallback/subscription propagation problem | `market-data-path-audit` | `diagnosing-trading-bugs` when a concrete defect must be reproduced and fixed |
| Strategy logic, score/threshold optimization, profitability claim, backtest or walk-forward review | `strategy-research-validation` | `tdd-trading-changes` only after a specific supported behavior change is selected |
| Live bot status, logs, deployed-SHA check, production degradation or incident diagnosis | `live-runtime-diagnosis` | `diagnosing-trading-bugs` only after production evidence identifies a reproducible code defect |
| Retired module, legacy alias, dead wiring, duplicate wrapper or compatibility cleanup | `architecture-cleanup` | `codebase-design` when ownership/interface redesign is actually required |
| Well-scoped behavior change with known owner | `tdd-trading-changes` | `runtime-contract-validation` or `codebase-design` only if the change crosses those concerns |
| Ownership, SSOT, module/interface or duplicate-path change | `codebase-design` | `tdd-trading-changes` once the design is resolved |
| External/broker/config/cross-module payload change | `runtime-contract-validation` | `tdd-trading-changes` for implementation |
| Machine-readable contract, replay fixture, property invariant, or historical regression benchmark | `runtime-contract-validation` | `strategy-research-validation` when strategy/backtest semantics are involved |
| Fuzzy or safety-critical proposal | `grill-trading-plan` | `domain-modeling-trading` only when terminology/state ownership is genuinely unclear |
| Durable requirements or multi-PR decomposition explicitly needed | `to-prd-trading-change` / `to-issues-trading-change` | Do not use for a narrow bug fix |
| PR/diff review | `pre-merge-trading-review` | none by default |
| End-of-session handoff | `session-worklog` | none |

**Do not automatically run the full skill chain.** Skills are selective procedures, not mandatory phases. Repository architecture and safety invariants remain authoritative in `AGENTS.md`.

## Pre-edit contract

For non-trivial edits, establish only what is needed to constrain the change:

```text
Objective:
Owner module:
Observable behavior:
Safety invariant:
Non-goals:
Focused validation:
```

Add rollback, interface, state-transition, or deployment details only when the change requires them.

## Validation

Generate a focused validation plan:

```bash
python scripts/agent_check.py --files path/to/changed.py
```

Execute the risk-aware fast validation ring:

```bash
python scripts/agent_check.py --files path/to/changed.py --run focused
```

For changes related to a known historical regression, select the smallest relevant benchmark case:

```bash
python scripts/agent_benchmark.py --case <case-id> --run
```

Execute focused checks plus the complete suite before merge when the environment supports it:

```bash
python scripts/agent_check.py --files path/to/changed.py --run full
```

Preserve the canonical runtime path. Never weaken risk, execution, instrument, or readiness safeguards merely to make a test or trade pass.
