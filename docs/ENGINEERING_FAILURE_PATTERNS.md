# Engineering failure patterns and agent memory

This document is the repository's durable memory for **recurring engineering mistakes** and the fastest way to prevent them.

It is not an incident log and it does not replace `AGENTS.md`, architecture ownership, or specialist skills. Keep only patterns that are repeated, expensive, safety-relevant, or easy for coding agents to reintroduce.

## How agents should use this

Before editing Python:

1. Read only the pattern sections relevant to the files being changed.
2. Apply the prevention rule while coding rather than repairing the issue after CI.
3. Run the earliest detector before broad tests.
4. If CI reveals a genuinely new recurring pattern, fix the code first; update this register only after the root cause is understood.
5. Prefer an executable guard or regression test over prose when the pattern can be checked automatically.

Do not add one-off typos, transient external failures, credentials, production account data, or speculative lessons.

## Coding-quality patterns

### FMT-001 — Black layout drift

**Failure signature**

- CI reports that Black would reformat a touched file.
- Follow-up commits contain only wrapping, comprehension, call, assertion, or blank-line changes.

**Why it recurs**

Agents often hand-format long expressions to look visually compact. Black owns layout and may choose a different canonical form.

**Prevention**

- Let Black own Python layout.
- Do not manually fight Black wrapping.
- Avoid mixing functional edits with unrelated formatting cleanup.
- For a new or previously Black-clean file, keep the whole file Black-clean.
- For legacy files, do not introduce Black changes on newly touched lines.

**Earliest detector**

Use the repository delta-aware checker through `scripts/agent_check.py --run focused`. When diagnosing directly, use `scripts/check_changed_black.py` against the branch base.

### LINT-001 — Ruff import, unused-name, and style regressions

**Failure signature**

- New Ruff `E`, `F`, or `I` diagnostics.
- Repeated follow-up fixes for import order, unused imports, line layout, or late-import exemptions.

**Why it recurs**

Incremental edits leave temporary imports behind, manually rebuild import blocks, or move code without re-checking the changed file.

**Prevention**

- Reuse existing imports instead of adding duplicates.
- Remove temporary imports after the implementation settles.
- Keep imports canonical and let Ruff/isort-compatible rules decide ordering.
- Preserve an existing late-import exemption only when the runtime ordering is intentional; do not add `noqa` merely to silence CI.
- Do not rewrite unrelated legacy lint debt in a narrow PR.

**Earliest detector**

Run the delta-aware Ruff checker before tests or PR publication.

### TYPE-001 — Strict mypy regressions on production code

**Failure signature**

- A changed production file introduces a new mypy error.
- Common causes include implicit `Any`, optional-value misuse, incompatible containers, or insufficient narrowing.

**Prevention**

- Reuse existing repository types and public interfaces.
- Type new state explicitly when inference is ambiguous.
- Narrow `None` and union cases before use.
- Prefer a precise local type over broad `Any`, blanket `cast`, or `type: ignore`.
- Do not change runtime behavior only to satisfy typing.

**Earliest detector**

Use the repository delta-aware mypy checker. It intentionally rejects **new** type debt while tolerating unrelated historical debt.

### SYNTAX-001 — Partial edits break syntax or import blocks

**Failure signature**

- malformed import groups;
- invalid indentation, brackets, or function boundaries;
- a follow-up commit only restores syntactically valid code after a multi-step edit.

**Prevention**

- Read the complete function/import block before replacing part of it.
- Prefer one coherent edit over many overlapping textual edits to the same lines.
- After structural/import edits, compile immediately before running expensive tests.

**Earliest detector**

```bash
python -m compileall -q src dashboard scripts
```

### SCOPE-001 — Narrow fixes inherit or create unrelated formatting debt

**Failure signature**

- a small functional PR becomes dominated by style churn;
- fixing one formatter/lint issue causes unrelated files or old lines to be reformatted;
- review can no longer distinguish behavioral changes from cleanup.

**Prevention**

- Treat the merge base as the quality baseline.
- Use the delta-aware Ruff/Black/mypy scripts rather than repository-wide cleanup during a focused correction.
- Do not "clean the file while here" unless cleanup is the explicit task.
- If legacy debt blocks a legitimate change, isolate the minimum touched-line cleanup and document it.

**Earliest detector**

Review `git diff --stat` and the changed-file quality output before broad tests.

### TEST-001 — Tests patch private or slotted implementation details

**Failure signature**

- monkeypatching an instance method/attribute fails because the object is slotted or immutable;
- tests pass only by reaching into private state that production callers do not use.

**Prevention**

- Test observable behavior through the public seam.
- Patch an existing adapter or class-level seam when instance mutation is not supported.
- Do not add production mutability solely to make a test easier.
- Keep mocks structurally similar to the production path.

**Earliest detector**

A focused test should fail for the intended business behavior, not because the test harness cannot mutate internals.

### TEST-002 — Wall-clock, timing, and order-sensitive tests

**Failure signature**

- a test passes alone but fails in the full suite;
- market/session state or asynchronous ordering changes the result;
- failures depend on current clock time rather than fixture time.

**Prevention**

- Use explicit timestamps and timezone-aware fixtures.
- Keep simulated market/session state deterministic.
- Avoid shared mutable state across tests.
- For suspected intermittents, reproduce alone and in the surrounding suite before editing production code.
- Do not hide a real regression by labelling it flaky without evidence.

**Earliest detector**

Focused test → adjacent suite → full suite. Compare the failure signature across all three.

### GIT-001 — Validated head becomes stale when main advances

**Failure signature**

- CI passed on a branch whose base is no longer current;
- a PR becomes non-mergeable or requires a last-minute rebuild;
- validation evidence no longer corresponds to the exact commit being merged.

**Prevention**

- Branch from the authoritative current `main`.
- Before merge, confirm the PR head is unchanged and `main` still matches the validated base.
- If `main` advances, rebuild/rebase safely and rerun final-head CI.
- Merge only the exact head that passed required checks.

**Earliest detector**

Run `scripts/agent_merge_guard.py --validated-base <sha> --validated-head <sha>` immediately before merge. It fails closed when the target base or validated head moved.

## Recurring bot-engineering patterns

These are routing fingerprints, not duplicated architecture specifications. Follow the linked owner/skill for the detailed procedure.

### DATA-001 — Fresh FULL quote/depth is downgraded or overwritten

Use `market-data-path-audit`.

Typical fingerprint: WebSocket FULL data exists upstream, but downstream sees LTP-only, stale, polling, or partial depth. Fix the earliest incorrect boundary; do not add another cache or fallback.

### STRAT-001 — Strategy result is contaminated by time identity or research leakage

Use `strategy-research-validation` and, for a concrete defect, `diagnosing-trading-bugs`.

Typical fingerprint: same-bar duplication, historical recovery using information unavailable at that bar, overlapping validation windows, or parameter choice justified only by in-sample P&L.

### ARCH-001 — Duplicate owner, monkey patch, wrapper, or compatibility path

Use `architecture-cleanup` and `codebase-design` only when the surviving interface must change.

Typical fingerprint: two modules appear to own the same contract, state, order path, history, or runtime behavior. Prove which path is active before deletion.

### ORDER-001 — Local order state is treated as broker execution truth

Use `diagnosing-trading-bugs` plus `tdd-trading-changes`.

Typical fingerprint: acknowledgement/request quantity is interpreted as a fill, partial/rejected/cancelled state becomes success, or bracket state diverges from reconciled position state.

### OBS-001 — Vague diagnostics cause speculative fixes

Use `live-runtime-diagnosis`.

Typical fingerprint: logs say only "failed", "not ready", or "no signal" without stage, symbol/token, source/freshness, blocker, or broker result. Improve the owning diagnostic seam before guessing.

## Canonical preflight for changed Python

The normal agent path is:

```bash
python scripts/agent_check.py --files <changed files> --run focused
```

For the repository itself, focused agent validation should run changed-file Ruff/Black/mypy checks first, then compile and focused tests. Full validation remains:

```bash
python scripts/agent_check.py --files <changed files> --run full
```

GitHub final-head CI remains authoritative before merge.

## Controlled learning from CI

Failed CI runs are summarized by `.github/workflows/failure-memory-candidates.yml`. The workflow feeds failed log lines to `scripts/agent_failure_learn.py`, which only reports known-pattern candidates. It never edits this document automatically.

Use repeated candidates as evidence to add or strengthen an executable guard first. Promote a new prose pattern only after root-cause review.

## Maintaining this memory

Add or change a pattern only when at least one of these is true:

- the same failure has recurred;
- the failure caused meaningful review/CI delay;
- it can affect trading safety or production correctness;
- it is easy for an AI coding agent to reintroduce.

Every pattern should contain:

```text
stable ID
failure signature
root mechanism
prevention rule
earliest detector
owner/skill when applicable
```

When a pattern becomes impossible because an automated guard fully prevents it, keep the guard and shorten or retire the prose entry. The goal is less repeated reasoning over time, not an ever-growing handbook.
