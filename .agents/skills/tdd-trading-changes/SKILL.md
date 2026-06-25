---
name: tdd-trading-changes
description: Build or modify NIFTY scalper behavior test-first. Use for strategy, market-data, readiness, risk, execution, Telegram, or configuration changes where observable behavior and regression protection matter.
---

# Test-Driven Trading Changes

Read `AGENTS.md` first. Preserve the authoritative runtime path and module ownership.

## Core rule

Test observable behavior through the module's public interface. Avoid tests coupled to private methods, incidental data structures, or implementation order.

A test should remain valid after an internal refactor that preserves behavior.

## Work in vertical slices

Do not write a large batch of imagined tests and then a large implementation. Use one behavior at a time:

```text
RED: one focused behavior test fails
GREEN: minimum safe code makes it pass
REFACTOR: simplify only while all tests are green
```

Each cycle must produce a usable, reviewable increment.

## Before coding

Define:

- the public interface being changed
- the exact observable behavior
- the safety invariant that must remain true
- the module that owns the behavior
- the files that must not be touched
- the validation command that proves the slice

Do not create a new seam merely to make mocking easier. Use an existing seam or introduce one only when behavior genuinely varies across at least two adapters.

## Mandatory invariant coverage

Select the relevant invariants for every change:

### Instrument and data

- NIFTY spot is context-only.
- Futures are context-only when enabled.
- Only resolved NIFTY option instruments are executable.
- Quote freshness and source are preserved.
- FULL-depth data is not silently downgraded to LTP-only.
- Contract selection remains owned by `InstrumentManager`.
- History remains owned by the market-data manager.

### Evaluation

- A completed bar is not evaluated twice unless explicitly designed.
- Stale context cannot silently override fresh option data.
- Rejection and readiness blockers are specific and observable.
- Signal output includes symbol, direction, score, entry, stop, target, reasons, and blockers.

### Risk and execution

- Risk, capital, cooldown, open-position, and max-loss guards remain active.
- Duplicate order events are idempotent.
- Rejected, timed-out, or partially filled orders cannot be reported as clean success.
- Broker acknowledgement and local position state reconcile correctly.
- Paper, shadow, and live modes remain separated.

## Preferred test levels

Use the shallowest level that still exercises the real behavior:

1. Pure unit test for deterministic calculations and decisions.
2. Module-level test through the public interface.
3. Integration test across an actual seam, using a controlled adapter.
4. Fixture replay for ticks, candles, broker responses, reconnects, and order updates.
5. End-to-end dry run without live order placement.

Avoid broad mocks that make the tested path different from production.

## Per-cycle checklist

- [ ] Test name states business behavior.
- [ ] Test fails for the intended reason before the code change.
- [ ] Test uses public behavior, not private implementation details.
- [ ] Minimal code passes the current test.
- [ ] No unrelated refactor or speculative feature is included.
- [ ] Relevant negative and failure-path behavior is tested.
- [ ] Risk and execution guards are asserted when the path can trade.
- [ ] The test cannot place a live order.

## Refactoring rule

Never refactor while red. After all relevant tests pass:

- remove duplication
- deepen modules where it reduces caller knowledge
- simplify interfaces
- preserve repository ownership boundaries
- run focused tests after each refactor step

## Completion

Run the validation required by `AGENTS.md`, including at minimum:

```bash
python -m compileall -q src
pytest -q
```

When the complete suite cannot run, report exactly what ran, what failed, why it failed, and whether the failure is related to the change.

This workflow is adapted from Matt Pocock's MIT-licensed `tdd` skill and tailored to this repository.