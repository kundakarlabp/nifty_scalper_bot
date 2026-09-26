---
name: architecture-cleanup
description: Safely remove retired modules, legacy aliases, duplicate wrappers, dead runtime wiring, compatibility paths, and obsolete architecture from the NIFTY scalper without deleting still-active behavior or creating replacement complexity.
---

# Architecture Cleanup

Use this skill for cleanup and simplification, not for ordinary localized bug fixes. Read `AGENTS.md`, `docs/REPO_MAP.md`, and `codebase-design` principles first.

## Goal

Delete proven-obsolete complexity while preserving the canonical runtime path and public behavior.

A smaller diff is preferable to a broad rewrite. Do not replace one legacy layer with a new abstraction unless the current owner genuinely needs it.

## Prove inactivity before deletion

For every candidate module, alias, wrapper, or path, inspect:

- direct imports
- re-exports and package `__init__` files
- constructor/wiring sites
- dependency-injection registration
- string/dynamic imports
- callbacks and plugin registries
- config/environment references
- CLI/entry points
- deployment/service scripts
- persistence/restart/recovery code
- tests and fixtures
- docs/operator commands
- external/shared interface references where applicable
- recent git history when it helps explain why the path exists

Do not infer dead code merely because a simple text search has no call expression.

## Classify the candidate

- `ACTIVE` — participates in current runtime or supported interface
- `COMPATIBILITY_REQUIRED` — not internally active but still supports a real external/import contract
- `TEST_ONLY` — intentionally exists only for test/simulation support
- `RETIRED` — no supported runtime, external, deployment, persistence, or test responsibility remains
- `UNKNOWN` — evidence is insufficient; do not delete

## Cleanup sequence

1. Identify the canonical owner that remains.
2. Prove the candidate has no unique responsibility.
3. Add or strengthen an architecture regression guard when recurrence is plausible.
4. Remove one coherent retired path.
5. Update imports/docs/tests that exist only because of that path.
6. Run focused architecture/affected-area tests.
7. Run full validation before merge.
8. Review the diff for accidental behavior changes.

## Compatibility rule

Keep a compatibility shim only when a current supported consumer actually needs it. Document that consumer and the removal condition.

Do not retain aliases indefinitely “just in case,” and do not remove a shim solely to make the tree look cleaner.

## High-risk cleanup

Use extra caution for:

- OrderManager / SafeOrderManager
- BracketManager and recovery state
- MarketDataManager/DataHub duplication
- broker adapters
- app startup wiring
- persisted state schemas
- deployment/health endpoints
- strategy registration

For these, require a public-interface regression or architecture test before deletion where practical.

## Completion report

```markdown
Removed:
Canonical owner retained:
Evidence the removed path was inactive:
Compatibility impact:
Regression guard:
Validation:
Residual risk:
```
