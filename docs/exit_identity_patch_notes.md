# Exit identity safety patch

This patch enforces the production invariant that protective exits remain exits throughout order submission and state accounting.

## Invariants

- Protective SL/TP/trailing/forced market orders carry `intent=EXIT`.
- Exit orders carry `linked_entry_order_id`, `trade_lifecycle_id`, and `bracket_id` when an owning bracket exists.
- In live mode, the native unresolved-exit gate treats immutable intent as the primary authority. Tag text is not sufficient to prove protective status.
- PositionManager hot-path symbol lookups canonicalize bare NIFTY option symbols to their `NFO:` form.
- Bare/canonical alias collisions are collapsed without summing quantities.

## Scope

This is a focused first-slice repair. It intentionally does not centralize all broker reconciliation or orphan adoption authorities; those should remain a separate PR.
