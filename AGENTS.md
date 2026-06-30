

````markdown
# AGENTS.md

## Project identity

This repository is a Python-based NIFTY options scalping bot.

It integrates:

- Zerodha/Kite broker APIs
- live market data
- WebSocket tick streaming
- option-chain / strike resolution
- NIFTY spot and optional futures context
- NIFTY option trading
- strategy scoring and gating
- risk management
- order execution
- Telegram diagnostics/control
- deployment/runtime health checks

The core objective is **stable, safe, high-confidence NIFTY options trading**, not frequent trading.

---

## Absolute architecture rules

These rules are non-negotiable.

### Instrument roles

- **NIFTY spot** is for direction/context only.
- **NIFTY futures** may be used only for volume/context if explicitly configured.
- **NIFTY options are the only tradable instruments.**
- Never place orders on NIFTY spot.
- Never place orders on NIFTY futures.
- Every executable trade must resolve to a valid NIFTY option instrument.

### Authoritative runtime path

Preserve this runtime flow unless explicitly instructed otherwise:

```text
core/app.py
→ data/market_data_manager.py
→ data/data_hub.py
→ strategies/runner.py
→ execution/order_manager.py
→ execution/bracket_manager.py
→ notifications/telegram_controller.py
````

Do not bypass this flow with side-channel patches.

---

## Module map (one line per major file)

Each core runtime file below carries a standardized module docstring (Runtime
role / Position in the pipeline / Owns–does NOT own / Safe-edit notes). See
`docs/REPO_MAP.md` for the fuller map.

### Core runtime path
| File | Purpose | Owns | Does NOT own |
| --- | --- | --- | --- |
| `core/app.py` | Orchestrator; builds context, wires subsystems, runs startup, computes readiness/arming SSOT | wiring, startup ordering, readiness SSOT | contract selection, history, order placement |
| `data/market_data_manager.py` | Sole owner of ticks + OHLC history; broker fetch; tick fan-out | tick cache, OHLC history, history fetch | contract selection, strategy logic |
| `data/data_hub.py` | Read facade over MDM (quotes/OHLC/OI/context) | read/derive helpers | history (none), contract selection |
| `strategies/runner.py` | Evaluation loop; gates; signal→order handoff | eval loop, exec state, runner/indicator history | contract selection, broker fetch, placement |
| `execution/order_manager.py` | THE live order path: placement, retries, lifecycle | order placement + lifecycle of record | signal generation, contract selection |
| `execution/bracket_manager.py` | Virtual SL/TP, ATR trailing, multi-target, partial scaling | bracket state, exit-fire decision | entry decisions, raw placement |
| `notifications/telegram_controller.py` | Operator console: commands, auth, diagnostics, alerts | handler registration, single-chat auth, messaging | trading decisions, order placement |

### Key support modules
| File | Purpose |
| --- | --- |
| `core/instrument_manager.py` | Authoritative contract selection + token resolution from the instrument dump |
| `instruments/active_contracts.py` | Canonical symbol helpers; active NIFTY future resolution |
| `execution/safe_order_manager.py` | Safety wrapper (guards/idempotency) around OrderManager |
| `execution/position_manager.py` | Position + pending-order state of record |
| `execution/readiness.py` | Pure readiness/arming decision helpers |
| `execution/order_executor.py` | Separate non-live executor — NOT the live order path |
| `core/strategy_manager.py` | Scores strategies and allocates between them |
| `strategies/signal_generator.py` | Produces scored signals from indicators |
| `core/market_regime.py` | Market-regime detection and fan-out |
| `data/rest/zerodha_client.py` | Low-level Kite REST + websocket client |
| `streaming/websocket_manager.py` | Hardened KiteTicker streaming |
| `risk/risk_manager.py` | Risk guardrails and telemetry |
| `config/settings.py` | Runtime settings facade |
| `backtesting/backtest_engine.py` | Event-driven historical replay with simulated fills/costs |

### SSOT invariants
- History is owned only by MDM; DataHub stores none. Never gate readiness on DataHub bars.
- Readiness gates on mdm/runner/indicator counts via the canonical functions in `core/app.py`.
- Contract selection lives in InstrumentManager; MDM/DataHub/runner consume, never select.


---

## Market-data architecture

### Required live-data flow

The intended live path is:

```text
startup
→ load instruments
→ resolve NIFTY spot
→ select CE/PE option basket
→ resolve option tokens
→ subscribe selected option tokens
→ receive WebSocket ticks
→ preserve bid/ask/depth if available
→ hydrate option OHLC
→ evaluate selected option symbols
→ pass only valid option signals to risk/execution
```

### Runtime overload invariants

- Never perform full-state persistence on every tick.
- Never schedule one coroutine/Future per tick from the WebSocket callback.
- Never silently lose open-position, selected-option, NIFTY spot, or active-futures ticks.
- Do not hold producer locks while processing candles, sorting large batches, resolving symbols expensively, or dispatching callbacks.
- MarketDataManager owns bounded tick ingress and candle construction.
- DataHub owns read-facade state and bounded quote/order/position snapshots only.
- `/livez` is liveness only, not trading readiness.
- Blocked readiness must always include a precise primary blocker.

### WebSocket and polling rules

* WebSocket should be the primary source of live ticks where available.
* Polling fallback must not silently replace full tick/depth data.
* If only LTP is available, mark it clearly as LTP-only.
* Preserve quote-quality metadata through the pipeline:

  * bid
  * ask
  * spread
  * depth
  * source / quote_source
  * timestamp
  * timestamp_ms
  * tradable_quote
  * stale/fresh state
* Do not downgrade FULL depth ticks into generic LTP-only ticks.
* Do not drop option ticks silently.

---

## Strict contract/data SSOT

- `core/instrument_manager.py` is the only live contract selector/cache for NIFTY spot, futures, options, symbol-token mappings, ATM CE/PE, and `ActiveContractBasket`.
- `data/market_data_manager.py` owns token subscriptions, quote/depth/OI state, polling fallback, and basket hydration.
- `data/candle_engine.py` owns tick-to-OHLC bars and bar readiness.
- `data/data_hub.py` is a read facade over the active basket and market data only.
- Strategies consume prepared context only; they must not select contracts, call broker instruments, or fetch historical data directly in the live loop.
- Duplicate selectors are compatibility wrappers only and must delegate to InstrumentManager or remain non-live/env-gated legacy fallback.
- Do not create new runtime selector files.
- Do not manually generate live NIFTY futures/options symbols.
- Do not derive futures from selected option month in live runtime.
- Do not silently ignore token, subscription, quote, depth, OI, or OHLC hydration failures.

---

## Readiness and hydration rules

Do not bypass readiness gates.

If readiness blocks trading, the bot must explain the exact blocker.

Examples of acceptable blocker reasons:

```text
spot_ltp_missing
spot_stale
option_token_missing
option_not_subscribed
option_tick_missing
option_ohlc_insufficient
option_quote_ltp_only
bid_ask_missing
spread_too_wide
depth_missing
same_bar_eval_skipped
strategy_gate_blocked
risk_gate_blocked
cooldown_active
position_already_open
execution_disabled
broker_unavailable
```

Avoid vague messages such as:

```text
not ready
readiness failed
no signal
gate blocked
```

unless they include a specific stage and reason.

---

## Strategy rules

* Do not add more indicators unless the live data path is already proven stable.
* Do not optimize strategy scoring before fixing:

  * option subscription
  * option hydration
  * timestamp parsing
  * quote-quality propagation
  * readiness diagnostics
  * execution safety
* Do not allow stale spot/futures context to veto fresh option signals unless explicitly intended and logged.
* Do not allow stale cached scan context to override fresh live option data without a clear diagnostic reason.
* Do not generate trades from spot-only signals.
* Strategy outputs must clearly identify:

  * symbol
  * instrument type
  * direction
  * confidence/score
  * entry
  * stop-loss
  * target
  * reasons
  * blockers if rejected

---

## Execution safety rules

Execution code is safety-critical.

Do not modify order execution unless the task explicitly requires it.

When modifying execution-related files:

* Never bypass risk checks.
* Never bypass capital checks.
* Never bypass open-position checks.
* Never bypass cooldowns.
* Never bypass max-loss / shutdown logic.
* Never place trades on spot or futures.
* Never place orders without a resolved option instrument token.
* Never hide broker errors with broad exception handling.
* Never return success after a failed broker operation unless the operation is explicitly confirmed safe.

Execution changes must include logs/tests showing:

```text
symbol
instrument type
quantity
entry
stop-loss
target
order side
mode: live/paper/shadow
risk status
broker response
```

---

## `.env` and configuration rules

Important repository-specific instruction:

* `.env` is intentionally present for current runtime tuning.
* Do not delete `.env`.
* Do not remove `.env` from the repo.
* Do not add `.env` to `.gitignore` or `.dockerignore` unless explicitly instructed by the user.
* Do not modify `.env` keys unless the task explicitly requires configuration correction.
* Do not add Zerodha access tokens, broker access tokens, or broker secrets into `.env` unless explicitly instructed by the user.
* Do not print secrets in logs.
* Do not hardcode credentials in source code.
* Keep `.env.example` dummy-only if created.

Configuration must remain centralized and explicit. Do not introduce hidden defaults that change live-trading behavior silently.

---

## Coding rules for agents

### Do not perform broad rewrites

Avoid:

* project restructuring
* path restructuring
* renaming modules
* renaming public classes/functions
* replacing large subsystems
* creating duplicate helper modules
* “clean architecture” rewrites without explicit instruction

Prefer:

* minimal code-anchored corrections
* small focused PRs
* existing module ownership
* tests proving the exact bug is fixed

### No monkey patches

Do not add code that only suppresses symptoms.

Bad patterns:

```python
try:
    ...
except Exception:
    pass
```

```python
if error:
    return True
```

```python
# temporary fallback
symbol = "NIFTY"
```

```python
# ignore readiness and continue
```

### Exception handling

* Avoid broad `except Exception` in critical paths.
* If broad exception handling already exists, do not expand it.
* When touching critical paths, narrow exceptions where practical.
* Always log actionable context:

  * module
  * function
  * symbol
  * token
  * stage
  * exception type
  * reason

### Public interfaces

Do not rename or change these without updating all call sites and tests:

* public classes
* public methods
* config keys
* environment variable names
* broker adapter interfaces
* strategy output schema
* runner signal schema
* order manager input schema
* Telegram command names

---

## PR scope rules

Every PR must be narrow.

Allowed PR types:

```text
guardrail/ci
investigation-only
market-data
hydration
timestamp/same-bar
quote-depth
readiness-diagnostics
risk
execution-safety
telegram-diagnostics
strategy-scoring
deployment
```

Do not mix unrelated categories in one PR.

Examples:

* Do not change execution while fixing market-data hydration.
* Do not change strategy scoring while fixing timestamp parsing.
* Do not change Telegram UI while fixing order execution.
* Do not refactor app startup while fixing one readiness gate unless required and justified.

---

## Mandatory workflow for Codex/agents

### Step 1 — Investigate before editing

For non-trivial bugs:

1. Inspect relevant files.
2. Map the call chain.
3. Identify exact root cause.
4. Identify affected files.
5. Identify files that must not be touched.
6. Propose minimal fix.
7. Only then edit.

Do not jump directly to edits for complex runtime bugs.

### Step 2 — Make minimal corrections

* Modify the smallest safe set of files.
* Preserve existing architecture.
* Preserve runtime behavior unless intentionally fixing it.
* Prefer tests over assumptions.

### Step 3 — Validate

Before marking any task complete, run at minimum:

```bash
python -m compileall -q src
pytest -q
```

If tests are missing for the changed area, add focused tests where practical.

If existing full test suite cannot run because of missing external services, document exactly:

```text
what was run
what failed
why it failed
whether failure is related to the change
```

Do not claim success if validation did not run.

---

## Recommended additional checks

Run these when relevant:

```bash
python -m compileall -q src
pytest -q tests/strategies
pytest -q tests/data
pytest -q tests/execution
pytest -q tests/risk
pytest -q tests/notifications
```

For startup/runtime changes, run any existing dry-run/startup command if available.

If no dry-run exists, do not invent live broker calls. Add a safe smoke test instead.

---

## Required PR description

Every PR must include:

````markdown
## Root cause

Exact cause of the bug, with file/function references.

## What changed

List changed files and summarize each change.

## What did not change

Explicitly state important untouched areas.

## Validation

Commands run:

- `python -m compileall -q src`
- `pytest -q`

Results:

```text
paste result summary
````

## Runtime impact

Explain impact on:

* startup
* market data
* option hydration
* strategy evaluation
* risk
* execution
* Telegram diagnostics

## Regression risk

List possible risks and how they were mitigated.

````

---

## Trading-specific success criteria

A change is not successful merely because code compiles.

For live data path changes, success requires:

```text
selected CE/PE option symbols are resolved
option tokens are available
option symbols are subscribed
option ticks are received
option OHLC is hydrated
bid/ask/depth quality is preserved when available
StrategyRunner evaluates the option symbol
rejection/block reason is explicit if no trade occurs
````

For execution changes, success requires:

```text
only options are tradable
risk checks are applied
order request is explicit
broker response is logged
failure does not look like success
safety brackets are not duplicated
cleanup does not suppress errors
```

---

## Known high-risk areas

Treat these files as high-risk. Do not modify them casually.

```text
src/nifty_scalper_bot/core/app.py
src/nifty_scalper_bot/data/market_data_manager.py
src/nifty_scalper_bot/data/data_hub.py
src/nifty_scalper_bot/streaming/websocket_manager.py
src/nifty_scalper_bot/strategies/runner.py
src/nifty_scalper_bot/execution/order_manager.py
src/nifty_scalper_bot/execution/bracket_manager.py
src/nifty_scalper_bot/notifications/telegram_controller.py
```

When modifying these files:

* read the relevant surrounding code first
* inspect all call sites
* avoid local-only fixes
* add diagnostics/tests when possible

---

## Do not merge automatically

Agents must not merge PRs into `main`.

The user should merge only after:

```text
CI passes
compileall passes
pytest passes or unrelated failures are documented
PR scope is narrow
no architecture contract is violated
runtime logs are acceptable
no new import/path/config errors are introduced
```

---

## Preferred development order

Optimize the bot in this order:

1. Guardrails and CI
2. Live option data path
3. Option token resolution/subscription
4. Option OHLC hydration
5. Timestamp and same-bar evaluation
6. Quote depth and tradable quote propagation
7. Readiness diagnostics
8. Risk and execution safety
9. Telegram diagnostics
10. Strategy scoring
11. Profit optimization

Do not jump to profit optimization before data and execution correctness are stable.

---

## Final rule

When uncertain, do not patch blindly.

Return:

```text
I need to inspect these files/calls first:
- file A
- file B
- function C
- function D
```

Then investigate before editing.

```
```
