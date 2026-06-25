---
name: codebase-design
 description: Design or improve modules, interfaces, seams, and adapters in the NIFTY scalper. Use for architecture decisions, testability problems, tangled ownership, duplicate runtime paths, or proposals to restructure code.
---

# Codebase Design for the NIFTY Scalper

Read `AGENTS.md` and `docs/REPO_MAP.md` before proposing structural changes. Preserve the authoritative runtime path unless the user explicitly requests an architecture change.

## Vocabulary

Use these terms consistently:

- **Module:** a function, class, package, or runtime slice with an interface and implementation.
- **Interface:** everything callers must know, including signatures, invariants, ordering, errors, configuration, and performance constraints.
- **Implementation:** behavior hidden behind the interface.
- **Seam:** a location where behavior can vary without editing the caller.
- **Adapter:** a concrete implementation placed at a seam.
- **Depth:** useful behavior delivered per unit of interface knowledge.
- **Locality:** keeping change, knowledge, bugs, and verification concentrated in the owning module.

## Target shape

Prefer deep modules: small interfaces that hide substantial behavior. Avoid shallow pass-through layers that merely rename or forward calls.

Ask:

- Can callers know less?
- Can parameters or return types be simpler?
- Can invariants be enforced once in the owner rather than repeated in callers?
- Can tests exercise the same interface used by production callers?
- Would deleting this module spread its complexity across several callers? If not, it may not be earning its place.

## Repository ownership

Respect these established seams:

- `InstrumentManager`: contract selection and symbol/token resolution.
- `MarketDataManager`: ticks, subscriptions, quote quality, and OHLC history.
- `DataHub`: read facade only.
- `StrategyRunner`: evaluation loop and signal-to-order handoff.
- `RiskManager`: risk decisions and telemetry.
- `OrderManager`: live order placement and lifecycle.
- `BracketManager`: protective exit state and decisions.
- `PositionManager`: position and pending-order state of record.
- broker clients: external broker adapters.
- Telegram controller: operator interface and diagnostics, not trading decisions.

Do not create duplicate selectors, parallel order paths, hidden history stores, or strategy-owned broker access.

## Seam rules

- Accept dependencies rather than constructing them inside business logic.
- Return explicit results rather than mutating distant state when practical.
- Introduce a seam only when behavior genuinely varies; one adapter is usually hypothetical, two adapters make the variation real.
- Internal seams may exist for implementation testing, but they must not enlarge the external interface.
- The production interface should also be the primary test surface.

## Trading-specific design checks

For every proposed interface, define:

- input freshness and timestamp semantics
- instrument type and token requirements
- failure modes and explicit blocker reasons
- idempotency expectations
- retry ownership
- state ownership and restart recovery
- paper/shadow/live behavior
- observability fields
- latency or throughput constraints

Do not hide these facts in implicit defaults.

## Design procedure

1. State the current ownership and call chain.
2. Identify the specific complexity leaking into callers.
3. Define the smallest useful interface at the correct seam.
4. Compare at least two plausible designs when the decision is consequential.
5. Evaluate each design for depth, locality, testability, failure transparency, and migration risk.
6. Choose the smallest safe change.
7. Add behavior tests through the interface.
8. Avoid broad restructuring in the same PR as a bug fix.

## Rejected patterns

- pass-through wrappers with no invariant or abstraction value
- a new module that duplicates an existing owner
- generic helper modules used as dumping grounds
- hidden fallback behavior that changes live trading silently
- architecture rewrites justified only as cleaner
- tests that must reach past the public interface to verify behavior

This workflow is adapted from Matt Pocock's MIT-licensed `codebase-design` skill and tailored to this repository.