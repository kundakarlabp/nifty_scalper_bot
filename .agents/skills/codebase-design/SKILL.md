---
name: codebase-design
description: Design or improve modules, interfaces, seams, and adapters in the NIFTY scalper. Use for architecture decisions, testability problems, tangled ownership, duplicate runtime paths, or restructuring proposals.
---

# Codebase Design for the NIFTY Scalper

Read `AGENTS.md` and `docs/REPO_MAP.md` before structural changes. Preserve the authoritative runtime path unless an architecture change is explicitly requested.

## Vocabulary

- **Module:** a function, class, package, or runtime slice with an interface and implementation.
- **Interface:** everything callers must know, including signatures, invariants, ordering, errors, configuration, and performance constraints.
- **Implementation:** behavior hidden behind the interface.
- **Seam:** a place where behavior can vary without editing the caller.
- **Adapter:** a concrete implementation placed at a seam.
- **Depth:** useful behavior delivered per unit of interface knowledge.
- **Locality:** keeping changes, bugs, knowledge, and verification in the owning module.

## Target shape

Prefer deep modules: small interfaces that hide substantial behavior. Avoid shallow pass-through layers.

Ask:

- Can callers know less?
- Can parameters or results be simpler?
- Can an invariant be enforced once in the owner?
- Can tests use the same interface as production callers?
- Would deleting the module spread complexity across callers? If not, it may be unnecessary.

## Repository ownership

Respect these established seams:

- `InstrumentManager`: contract selection and symbol/token resolution.
- `MarketDataManager`: ticks, subscriptions, quote quality, and OHLC history.
- `DataHub`: read facade only.
- `StrategyRunner`: evaluation loop and signal-to-order handoff.
- `RiskManager`: risk decisions and telemetry.
- `OrderManager`: order placement and lifecycle.
- `BracketManager`: protective exit state and decisions.
- `PositionManager`: position and pending-order state.
- broker clients: external broker adapters.
- Telegram controller: operator interface and diagnostics.

Do not create duplicate selectors, parallel order paths, hidden history stores, or strategy-owned broker access.

## Seam rules

- Accept dependencies instead of constructing them inside business logic.
- Return explicit results instead of mutating distant state when practical.
- Add a seam only when behavior genuinely varies.
- Internal seams must not enlarge the external interface.
- The production interface should be the primary test surface.

## Trading-specific interface checks

Define:

- timestamp and freshness semantics
- instrument type and token requirements
- explicit failure and blocker reasons
- idempotency and retry ownership
- state ownership and restart recovery
- paper, shadow, and live behavior
- observability fields
- latency or throughput constraints

Do not hide these in implicit defaults.

## Design procedure

1. State current ownership and call chain.
2. Identify complexity leaking into callers.
3. Define the smallest useful interface at the correct seam.
4. Compare at least two designs for consequential changes.
5. Evaluate depth, locality, testability, failure transparency, and migration risk.
6. Choose the smallest safe change.
7. Add behavior tests through the interface.
8. Keep architecture work separate from unrelated bug fixes.

## Reject

- pass-through wrappers with no invariant value
- modules duplicating an existing owner
- generic helper dumping grounds
- hidden fallbacks that silently change runtime behavior
- broad rewrites justified only as cleaner
- tests that must reach past the public interface

Adapted from Matt Pocock's MIT-licensed `codebase-design` skill for this repository.