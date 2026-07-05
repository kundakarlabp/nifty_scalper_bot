---
name: runtime-contract-validation
description: Validate external and cross-module contracts in the NIFTY scalper. Use when touching config, broker APIs, websocket/polling data, instrument dumps, strategy signals, order requests, broker acknowledgements, Telegram commands, fixtures, or replay/backtest inputs.
---

# Runtime Contract Validation

Use this skill to prevent malformed or stale external data from reaching strategy and execution logic.

## Boundary rule

All external or unstable input is unknown until validated.

Validate at these boundaries:

```text
.env / runtime settings
broker REST response
websocket tick
polling quote fallback
instrument dump row
active option basket
DataHub read result
strategy signal object
risk decision result
order request
broker acknowledgement
order update / fill event
Telegram command
backtest CSV / replay fixture
```

## Minimum contract fields

### Tick / quote

Require explicit handling of:

```text
symbol
instrument token
instrument type
ltp
bid
ask
spread
depth
source
timestamp
timestamp_ms
fresh/stale decision
LTP-only status
```

### Strategy signal

Require:

```text
symbol
instrument type
direction
score/confidence
entry
stop-loss
target
quantity intent if applicable
reasons
blockers
bar identity
timestamp
```

### Order request

Require:

```text
symbol
instrument token
instrument type
side
quantity
order type
product
mode: LIVE/PAPER/SHADOW
idempotency key
risk approval reference
source strategy
```

### Broker response

Require:

```text
status
order id if accepted
rejection reason if rejected
raw status classification
timestamp
retryability
position reconciliation requirement
```

## Failure behavior

Invalid input must result in one of:

```text
safe rejection
specific readiness blocker
specific risk blocker
operator diagnostic
retry with bounded policy
paper/shadow-only quarantine
```

Never:

- silently coerce unknown data into tradable data
- treat missing bid/ask as valid full quote
- infer an option token
- report failed broker operation as success
- use broad exception handling to hide contract failures

## Test requirements

For each boundary change, add or identify tests for:

- valid minimal payload
- missing required field
- wrong type or unit
- stale timestamp
- unsupported instrument type
- rejection/error payload
- duplicate or replayed event when relevant

## Implementation guidance

Prefer existing project conventions. In Python, use Pydantic models, dataclasses with explicit validation, enums, typed results, or narrow parser functions where appropriate.

Validation should happen at module boundaries, not scattered across callers.

## Output format

```markdown
## Contract reviewed
Boundary:
Owner module:
Input shape:
Validated fields:
Invalid-input behavior:
Tests required:
Files touched:
Residual risk:
```
