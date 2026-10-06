---
name: nifty-scalper-engineering
description: Automatically use for every engineering task involving kundakarlabp/nifty_scalper_bot or the NIFTY Scalper Bot, including code audit, code editing, debugging, refactoring, architecture understanding, repository review, strategy implementation review, validation, merge-readiness review, or requests to simplify/canonicalize the bot. This is the repository orchestration layer: apply the repository contract, load only the minimum relevant context, select one specialist repository skill, make minimal canonical changes, validate proportionally to risk, and report only verified states.
---

# Nifty Scalper Engineering — repository orchestrator

## Purpose

This is the default orchestration skill for engineering work in this repository.

Use it automatically whenever the task is to **understand, audit, review, edit, debug, refactor, validate, simplify, optimize, or merge** NIFTY Scalper Bot code. The user should not need to name this skill explicitly.

This skill does **not** replace the repository contract or specialist skills. It routes work into them without duplicating their detailed procedures.

## Authority and precedence

Use this order:

1. `AGENTS.md` — behavioral and safety contract.
2. `docs/architecture/agent_manifest.json` — machine-readable ownership, runtime path, high-risk markers, and validation routing.
3. This orchestration skill — task framing, minimal-context loading, specialist selection, and closure discipline.
4. One primary specialist skill from `docs/AGENT_START_HERE.md`; add another only when the task genuinely crosses that concern.
5. Current code, tests, CI, runtime evidence, and broker state — factual evidence.

If generic/global skill guidance conflicts with repository-local rules, repository-local rules win.

## Automatic activation

Activate this skill for any request concerning this repository that includes one or more of:

- audit, deep-dive audit, understand, review, inspect, trace, status;
- edit, fix, implement, correct, simplify, canonicalize, refactor, clean up;
- strategy, market data, scoring, gate, blocker, risk, order, broker, stop, position, backtest;
- test, Ruff, Black, mypy, syntax, CI, regression, validation;
- PR, diff, merge, deploy-readiness, production-readiness;
- "no loss of function", "no new errors", "no duplication", "no overengineering".

Do not wait for an explicit `$nifty-scalper-engineering` mention.

## Fast orchestration sequence

### 1. Recover exact state

- identify repository, branch/base/head, and existing changes;
- read `docs/AGENT_START_HERE.md` and `docs/REPO_MAP.md`;
- read full `AGENTS.md` before any high-risk runtime edit;
- use `python scripts/agent_task.py context --query "..."` when the owner/path is not already obvious;
- do not repeat broad repository discovery once access and ownership are known.

### 2. Classify blast radius

Use the smallest safe depth:

- **Local** — one owner, one behavior, low-risk surface: focused context and focused regression.
- **Cross-cutting** — multiple owners/contracts: trace interfaces and affected state.
- **Critical trading** — market data truth, risk, sizing, order lifecycle, reconciliation, protective exits, persistence, or live mode: full relevant trading-path and failure-path audit.

A "deep dive" means enough depth to prove the requested behavior and its safety effects, not reading every file.

### 3. Select one specialist skill

Route through `docs/AGENT_START_HERE.md`.

Typical mapping:

- concrete failure → `diagnosing-trading-bugs`
- WebSocket/depth/freshness → `market-data-path-audit`
- strategy/backtest/profitability → `strategy-research-validation`
- well-scoped code change → `tdd-trading-changes`
- duplicate/legacy cleanup → `architecture-cleanup`
- ownership/interface change → `codebase-design`
- boundary/schema/broker payload → `runtime-contract-validation`
- live status/incident → `live-runtime-diagnosis`
- PR/final diff review → `pre-merge-trading-review`

The orchestrator does not count as the specialist skill. Do not run the entire catalog.

### 4. Establish the patch contract

For non-trivial changes define:

```text
Objective:
Owner module:
Observable behavior:
Safety invariant:
Non-goals:
Focused validation:
```

Trace symptom → canonical owner → downstream safety effect.

Prefer deletion or modification of the existing owner over wrappers, parallel managers, compatibility layers, hidden fallback paths, or duplicated state.

### 5. Implement and verify

- reproduce before fixing when practical;
- add or strengthen a regression that can fail for the right reason;
- make the smallest owner-consistent change;
- preserve unrelated behavior;
- remove superseded paths end to end when behavior is intentionally retired;
- run `python scripts/agent_task.py check --files <changed files>`;
- repair introduced/exposed validation failures and rerun;
- run `python scripts/agent_task.py full --files <changed files>` before merge when supported;
- review the exact final diff and final-head CI.

For risky refactors, characterize current intended behavior first and use differential before/after checks where practical.

### 6. Merge discipline

Before merge:

- verify exact base/head;
- confirm no unrelated diff;
- confirm final-head validation/CI;
- resolve valid review findings;
- use `python scripts/agent_task.py merge-check --validated-base <base-sha> --validated-head <head-sha>`;
- never report "merged", "deployed", or "observed live" without direct evidence.

## Repository learning

This repository already has a durable engineering memory:

`docs/ENGINEERING_FAILURE_PATTERNS.md`

Use it instead of creating a parallel `.codex/NIFTY_SCALPER_LEARNINGS.md` diary.

When a recurring failure is verified and materially reusable:

1. fix the immediate defect;
2. identify the prevention rule or detector;
3. update the matching repository failure pattern only when warranted;
4. prefer executable prevention/tests over additional prompt text.

## Non-negotiables

- Preserve capital-protective fail-closed behavior.
- Do not weaken hard risk/execution/readiness safeguards merely to increase trades.
- Do not reintroduce retired scoring/legacy paths through aliases or fallbacks.
- Do not create multiple owners for the same state or calculation.
- Do not claim profitability from code quality or parameter changes alone.
- Do not claim "no loss of function" without relevant regression evidence.
- Never place real-money orders merely to validate a code change.

## Completion

For substantial work, report the exact verified state:

`understood → patched → focused-tested → full-tested → reviewed → merged → deployed → live-observed`

Only mark stages that were actually observed. State remaining blockers precisely.
