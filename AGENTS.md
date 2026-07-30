NIFTY Scalper Bot — Repository Guidance
Mission
Maintain a deterministic, observable, capital-protective Python 3.12+ NIFTY options trading system using Zerodha KiteConnect, live market data, Telegram controls, strategy gating, risk management, and automated execution.

Capital protection takes priority over trade frequency and apparent profitability.

Success means:

only valid NIFTY option contracts can reach execution;
market data and readiness states are explicit;
strategies consume prepared context;
risk and broker constraints cannot be bypassed;
failures and blocks are actionable;
changed behavior is covered by focused validation.
Do not optimize for guaranteed profitability. Strategy performance must be evaluated separately through reproducible backtesting, costs, slippage, out-of-sample testing, paper trading, and live metrics.

Product boundaries
Trade NIFTY options only.
NIFTY spot and futures are context only.
Never place spot or futures orders.
Every executable signal must resolve to a broker-validated NIFTY option symbol and instrument token.
Prefer fewer valid trades over noisy signals.
Do not weaken safety controls to increase trade count.
Source-of-truth ownership
Domain	Owner
Instrument discovery, contracts, symbol/token mapping	core/instrument_manager.py
Runtime basket commit	core/app.py
Subscription, quote, depth, OI and hydration state	data/market_data_manager.py
Tick-to-OHLC bars	data/candle_engine.py
Strategy-facing data facade	data/data_hub.py
Strategy orchestration and evaluation	core/strategy_manager.py, strategies/*
Risk policy and sizing	risk/*
Order construction and broker execution	execution/*
Telegram controls and diagnostics	notifications/*
Do not create competing selectors, instrument caches, contract generators, readiness owners, or execution paths.

Strategies must not:

select or generate contracts;
fetch broker instruments;
call broker historical data in the live evaluation loop;
bypass data, risk, or execution readiness.
Execution must not bypass:

execution mode;
instrument validation;
risk and daily-loss limits;
margin and lot-size checks;
open-position and cooldown checks;
SL/TP validation;
broker order-state confirmation.
Runtime flow
InstrumentManager selects validated basket
→ App commits active basket
→ MarketDataManager subscribes and hydrates
→ CandleEngine builds bars
→ DataHub exposes prepared context
→ StrategyManager evaluates option candidates
→ Risk validates and sizes
→ Execution submits option orders
WebSocket FULL data is primary. Polling is fallback and must not overwrite fresher FULL-depth data.

Preserve through the data path:

symbol, token, timestamp, timestamp_ms, bid, ask, spread,
depth, OI, source, freshness, stale state, tradable_quote
Do not evaluate a strategy until its explicitly required data is ready.

Missing optional context should reduce confidence, not automatically block trading, unless the active strategy declares that context mandatory.

Blockers and gates
Use the sequence:

data → strategy → risk → execution
A gate must protect an actual invariant and have one owner. Before adding a gate, check whether the same condition already exists or whether the upstream state should be fixed instead.

Every blocked candidate must expose:

stage
symbol
blocker_code
required_value
actual_value
recoverable
owner
Reuse existing blocker codes. Add a new code only when no current code accurately represents the condition.

A healthy runtime must not become a permanent no-trade system because of duplicated, contradictory, or nonessential gates.

Risk and order invariants
For a BUY order:

stop_loss < execution_or_fill_price < take_profit
Anchor SL/TP to the confirmed execution/fill price when required by the order workflow.

Position size is limited by both risk and margin:

risk_lots = risk_amount / risk_per_lot
margin_lots = available_margin / margin_per_lot
final_lots = min(risk_lots, margin_lots)
If fewer than one valid lot can be traded, skip and record the exact reason.

Do not enable live execution, modify credentials, or weaken production risk limits without explicit authorization.

Change discipline
Before a non-trivial edit:

read the closest applicable AGENTS.md;
trace the relevant runtime path and reproduce the problem when feasible;
identify the owner and affected interfaces;
define the smallest coherent change and regression test;
check whether NIMS-Chrome or another declared integration is actually affected.
Prefer existing owners and public interfaces. Do not add dependencies, helper modules, broad refactors, or compatibility layers unless necessary for the requested outcome.

Do not modify unrelated files. Preserve existing user changes.

When ownership or runtime behavior changes, update the appropriate architecture documentation. Use top-of-file role notes only where they clarify a non-obvious boundary; do not add repetitive boilerplate to every file.

Validated correction and publishing workflow
For requested code fixes, use the repository's established one-issue-at-a-time workflow:

1. Resolve the current authoritative `main` commit through GitHub and start from that exact commit in a clean branch/worktree. Do not edit a stale or dirty checkout, and do not include runtime artifacts such as `positions.json`, caches, logs, or generated data.
2. Deep-trace the failing path before editing: observed log/test symptom → entry point → owning component → downstream safety effect. Read the complete relevant functions and adjacent interfaces; distinguish the root code defect from data, environment, timing, and test-fixture failures.
3. Define one small behavioral invariant and first add or identify a regression that fails for the defect. Reuse the existing owner, interface, cache, configuration, and journal. Prefer a few coherent lines over a new helper/module, broad refactor, duplicated mechanism, weakened threshold, or unrelated optimization.
4. Validate in widening rings: focused regression, adjacent module/suite, production-file compilation, changed-file Ruff/Black/mypy where configured, then the complete repository suite. Preserve the exact commands and results for the PR.
5. Treat environment-only dependency failures and suspected timing/order-sensitive failures as evidence to investigate, not automatic code failures. Reproduce the failing test alone and rerun the remaining suite when justified. Never hide a genuine regression or describe an unrun suite as passing.
6. Commit and publish only the intended validated files. Confirm the remote branch is based on the authoritative `main` commit and that every remote blob matches the locally tested blob before opening the PR.
7. Open one focused PR with the root cause, behavior change, preserved behavior, and validation. Hold merge until all required GitHub workflows are successful, including full-suite, execution-safety, market-aware/E2E, and changed-code quality jobs when present.
8. Merge the unchanged validated head, verify the resulting `main` commit, then start the next correction from that merge. Do not stack several unmerged trading-path fixes.

GitHub access fallback:

- Local `git`/`gh` is useful but is not the only valid publishing path for this repository.
- If `gh` is missing, unauthenticated, or local push credentials are unavailable, first check the authenticated GitHub connector and repository permissions.
- When the connector has access, use its Git data and pull-request operations to create the branch, blobs/tree/commit, update the ref, open the PR, inspect workflow status/logs, and merge. Verify exact blob equality against the locally validated files.
- Do not stop or ask the user to install `gh` merely because the CLI path is unavailable while the authenticated connector can complete the work safely.
- Report a publishing blocker only after both the local GitHub path and the connector path are unavailable or require user authentication/authority. A dirty or stale checkout is not a blocker; preserve it and create a clean worktree from the authoritative commit.

Cross-repository integration
NIMS-Chrome is a dependent repository only when the requested behavior crosses a shared interface, configuration, schema, messaging protocol, deployment contract, or user workflow.

For such tasks:

trace the integration before editing;
identify the contract owner;
update all repositories required for a complete compatible change;
avoid speculative edits in unaffected repositories;
validate both sides of the boundary;
report repositories that were unavailable or not validated.
Error handling and observability
Never suppress failures or return false success.

At process boundaries, a broad exception is acceptable only when it provides:

stage and symbol/context;
actionable error details;
safe fallback or fail-closed behavior;
correct success/failure state.
Do not log secrets, tokens, credentials, or sensitive account data.

Validation
Run the smallest relevant checks first, followed by broader checks when proportionate to the change.

Standard checks:

python -m compileall -q src
pytest -q
Useful focused suites:

pytest -q tests/data
pytest -q tests/strategies
pytest -q tests/core/test_strategy_manager_context_to_option_propagation.py
pytest -q tests/execution tests/risk
Add regression coverage for recurring architectural failures, including:

no spot/futures execution;
no manual live contract generation;
no option-derived futures selection;
no broker-instrument access from strategies;
active-basket-only subscriptions;
no contract selection in DataHub;
explicit hydration blockers;
no bypass of risk or execution gates.
Report the commands actually run and their results. If a check cannot run, state why and perform the best available alternative.

Completion criteria
A change is complete when:

the requested behavior is implemented;
ownership remains unambiguous;
option-only execution is preserved;
failure behavior is explicit;
focused regression coverage exists;
relevant validation passes or the limitation is reported;
affected documentation and integration contracts are consistent.
Final implementation report:

Outcome:
Files changed:
Validation:
Cross-repository impact:
Remaining risk:
For diagnosis-only work, report prioritized findings and evidence without editing files.
