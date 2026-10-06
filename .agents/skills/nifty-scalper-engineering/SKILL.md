---
name: nifty-scalper-engineering
description: Automatically use for every engineering task involving this NIFTY Scalper Bot repository: audit, understand, review, debug, edit, refactor, simplify, validate, or merge-readiness work. This is a thin orchestration bridge to the repository contract and one task-specific specialist skill; it should reduce context and process overhead, not add another framework.
---

# Nifty Scalper Engineering

## Purpose

Default lightweight orchestrator for repository engineering. Do not treat it as another specialist procedure.

Authority:
1. `AGENTS.md`
2. `docs/architecture/agent_manifest.json`
3. current code/tests/CI/runtime evidence
4. this orchestrator
5. one primary specialist skill from `docs/AGENT_START_HERE.md`

If the account-level Nifty Scalper Engineering skill is already active, treat both as the **same orchestration layer**; do not duplicate their procedures in context.

## Fast path

1. **Recover exact state**
   - Confirm current `main`, branch/head, relevant diff, and current vs historical symptom.
   - Read `docs/REPO_MAP.md`; read full `AGENTS.md` before high-risk runtime edits.
   - Reuse verified context instead of repeating broad repository discovery.

2. **Scope by risk**
   - Local: one owner/behavior → focused path/test.
   - Cross-cutting: multiple owners/contracts → trace affected interfaces/state.
   - Critical trading: data, strategy routing, risk, sizing, orders, reconciliation, protective exits, persistence/live mode → inspect relevant normal + failure paths.

3. **Load one specialist**
   - Use `docs/AGENT_START_HERE.md`.
   - Add another specialist only when the task genuinely crosses concerns.
   - Read only matching entries from `docs/ENGINEERING_FAILURE_PATTERNS.md`.

4. **Fix the owner, not the symptom**
   - Define objective, canonical owner, observable behavior, safety invariant, non-goals, and focused validation.
   - Prefer modifying/deleting existing canonical code over wrappers, parallel managers, compatibility layers, hidden fallbacks, new dependencies, or duplicate state.

5. **Validate and repair**
   - `python scripts/agent_task.py check --files <changed files>`
   - `python scripts/agent_task.py full --files <changed files>` before merge when supported.
   - On syntax/Ruff/Black/type/test/architecture failure: classify → fix introduced/directly exposed issue → rerun failed gate and affected upstream checks.

6. **Close on exact evidence**
   - Review `base...head`, scope, duplicate/legacy remnants, accidental deletions, safety effects, and exact-head CI.
   - Reconcile base drift before merge.
   - Read back `main` after merge.

## Non-negotiables

- Preserve fail-closed behavior for ambiguous market data, broker/order state, risk state, and execution safety.
- Do not weaken hard risk/readiness/execution safeguards to increase trade count.
- Keep one source of truth; do not reintroduce retired scoring, duplicate managers, shadow state, or alternate execution paths.
- Prefer the smallest coherent change; avoid unrelated cleanup and overengineering.
- Never claim no loss of function without relevant regression evidence.
- Never claim profitability from code/refactor/parameter changes without realistic cost and out-of-sample/live evidence.
- Never place real-money orders merely to validate code.

## Completion

Report only observed states:

`understood → patched → focused-tested → full-tested → reviewed → merged → deployed → live-observed`

Stop expanding the audit once ownership, root cause, affected boundaries, and falsifying tests are clear.
