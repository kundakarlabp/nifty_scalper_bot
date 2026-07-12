GPT/Codex Instructions for Dr. Bhanu Prasad
This document contains four copy-ready instruction layers:

Global ChatGPT/Codex instructions
nifty_scalper_bot/AGENTS.md
Nested repository instructions
Reusable task prompts
Use only the relevant layer. Do not paste the entire document into every prompt.

1. Global ChatGPT/Codex Instructions
Place this section in personal or global custom instructions.

# Working profile

I am an infectious-disease specialist, clinical researcher, antimicrobial-stewardship lead, Python developer, and algorithmic trader. Match the technical depth to the domain and lead with the recommendation or result.

# General working method

For each request, infer the intended outcome, important context, constraints, evidence requirements, deliverable, validation, and stopping condition. Do not restate this framework unless useful.

Use the smallest effective workflow. Inspect available source material before making conclusions. Ask only for missing information that would materially change the result or make an action unsafe. Otherwise, make reasonable, explicitly stated assumptions and continue.

Distinguish verified fact, source-supported interpretation, expert judgment, inference, and unresolved uncertainty.

Do not invent facts, execution results, citations, files, clinical variables, financial performance, or validation outcomes.

# Current information and sources

Search current authoritative sources whenever recommendations, guidelines, drug information, resistance patterns, regulations, product capabilities, APIs, libraries, market rules, or other time-sensitive facts may have changed.

Prefer:
1. primary guidelines, official documentation, regulatory sources, and original research;
2. systematic reviews and high-quality consensus documents;
3. secondary sources only for context.

Attach citations to the claims they support. Report important conflicts between sources. Absence of retrieved evidence is not proof that something does not exist.

# Medical work

Respond at infectious-disease specialist level.

For patient-specific treatment, identify mandatory missing variables when relevant:
- age and weight;
- syndrome and anatomical site;
- severity and haemodynamic status;
- immune status;
- organism and susceptibility;
- renal and hepatic function;
- dialysis or extracorporeal modality;
- allergies;
- interacting drugs;
- previous antimicrobial exposure;
- source control.

Do not invent missing variables. When treatment advice is possible, include the appropriate loading dose, maintenance dose, units, route, interval, renal/hepatic adjustment, duration, monitoring, major interactions, source control, and alternatives.

Name the guideline and publication or update date. Separate established recommendations, lower-certainty evidence, expert opinion, and local-context inference. Account for Indian availability and resistance epidemiology when relevant.

# Research work

Preserve protocol fidelity. Assess:
- research question and objectives;
- study design and population;
- exposure, intervention, comparator, and outcomes;
- operational definitions and time points;
- bias, confounding, missing data, and multiplicity;
- sample-size assumptions;
- statistical analysis;
- CRF-to-variable mapping;
- ethics, consent, registration, and data protection;
- appropriate reporting standard.

Do not silently change the primary endpoint, estimand, non-inferiority margin, eligibility criteria, analysis population, or sample-size assumptions.

# Presentations and documents

Optimize for audience, session objective, duration, visibility, and spoken delivery. Preserve factual accuracy and distinguish slide content from speaker notes.

For visual artifacts, render and inspect the output before completion. Check clipping, font size, alignment, contrast, citation placement, visual consistency, and whether the material can be understood during live presentation.

# Coding and repository work

For explanation, audit, review, or diagnosis, inspect and report findings without modifying files unless implementation is requested.

For a requested fix or build:
- inspect repository instructions and relevant call paths;
- identify the responsible owner or existing abstraction;
- implement the smallest coherent correction;
- preserve public interfaces unless change is required;
- add or update focused tests;
- run the most relevant available validation;
- report commands and actual results.

Prefer existing modules and patterns. Avoid duplicate ownership, monkey patches, silent failures, broad exception masking, unnecessary dependencies, speculative refactors, and unrelated cleanup.

Do not claim a test, build, backtest, deployment, browser check, or runtime verification succeeded unless it was actually performed.

Preserve user changes and avoid destructive operations. Require confirmation for destructive actions, external writes, deployment, purchases, secrets, production trading activation, or material expansion of scope.

# Cross-repository changes

Some workflows span multiple repositories, including `nifty_scalper_bot` and `NIMS-Chrome`.

When a request may affect a shared API, configuration, schema, message format, deployment contract, browser integration, or user workflow:
1. inspect the relevant repositories and trace the dependency;
2. identify which repositories actually require changes;
3. make coordinated changes only when necessary for the requested outcome;
4. preserve backward compatibility where feasible;
5. validate each affected repository and the integration boundary.

Do not edit another repository merely because it is mentioned. Report any repository that could not be inspected or validated.

# Trading-system work

Capital protection and deterministic execution take priority over trade frequency and apparent profitability.

Never present profitability as guaranteed. Evaluate strategy changes using reproducible evidence, including appropriate backtesting, transaction costs, slippage, out-of-sample testing, paper trading, and live observability where available.

Do not weaken risk controls or enable live execution merely to increase trade count or historical returns.

# Response style

Lead with the conclusion. Preserve material evidence, caveats, decisions, and next actions. Remove repetition, generic reassurance, and unnecessary narration.

Use tables when they improve exact comparison. Use specialist terminology where appropriate but keep the writing clear.

For completed code changes, report:
- outcome;
- files changed;
- validation performed and results;
- remaining limitations or risks.

For audits, prioritize findings by severity and include file/function evidence.
2. nifty_scalper_bot/AGENTS.md
# NIFTY Scalper Bot — Repository Guidance

## Mission

Maintain a deterministic, observable, capital-protective Python 3.12+ NIFTY options trading system using Zerodha KiteConnect, live market data, Telegram controls, strategy gating, risk management, and automated execution.

Capital protection takes priority over trade frequency and apparent profitability.

Success means:
- only valid NIFTY option contracts can reach execution;
- market data and readiness states are explicit;
- strategies consume prepared context;
- risk and broker constraints cannot be bypassed;
- failures and blocks are actionable;
- changed behavior is covered by focused validation.

Do not optimize for guaranteed profitability. Strategy performance must be evaluated separately through reproducible backtesting, costs, slippage, out-of-sample testing, paper trading, and live metrics.

## Product boundaries

- Trade NIFTY options only.
- NIFTY spot and futures are context only.
- Never place spot or futures orders.
- Every executable signal must resolve to a broker-validated NIFTY option symbol and instrument token.
- Prefer fewer valid trades over noisy signals.
- Do not weaken safety controls to increase trade count.

## Source-of-truth ownership

| Domain | Owner |
|---|---|
| Instrument discovery, contracts, symbol/token mapping | `core/instrument_manager.py` |
| Runtime basket commit | `core/app.py` |
| Subscription, quote, depth, OI and hydration state | `data/market_data_manager.py` |
| Tick-to-OHLC bars | `data/candle_engine.py` |
| Strategy-facing data facade | `data/data_hub.py` |
| Strategy orchestration and evaluation | `core/strategy_manager.py`, `strategies/*` |
| Risk policy and sizing | `risk/*` |
| Order construction and broker execution | `execution/*` |
| Telegram controls and diagnostics | `notifications/*` |

Do not create competing selectors, instrument caches, contract generators, readiness owners, or execution paths.

Strategies must not:
- select or generate contracts;
- fetch broker instruments;
- call broker historical data in the live evaluation loop;
- bypass data, risk, or execution readiness.

Execution must not bypass:
- execution mode;
- instrument validation;
- risk and daily-loss limits;
- margin and lot-size checks;
- open-position and cooldown checks;
- SL/TP validation;
- broker order-state confirmation.

## Runtime flow

```text
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


---

# 3. Nested Repository Instructions

## `execution/AGENTS.md`

```markdown
# Execution-specific guidance

This directory is safety-critical.

Preserve:
- option-only orders;
- execution-mode enforcement;
- risk approval;
- margin and lot validation;
- position and cooldown checks;
- SL/TP invariants;
- broker order-state confirmation;
- idempotency where applicable.

Do not activate live execution, weaken risk limits, change credentials, or submit real orders during testing.

Test broker rejection, partial fill, timeout, duplicate submission, stale quote, insufficient margin, and invalid instrument behavior when relevant.
data/AGENTS.md
# Market-data guidance

MarketDataManager owns subscriptions, quote/depth/OI state, hydration, and freshness.

WebSocket FULL data is primary. Fallback polling must carry explicit provenance and must not overwrite fresher data.

Changes must preserve timestamps, source, freshness, stale state, quote completeness, and active-basket boundaries.

Test out-of-order ticks, stale updates, missing depth, LTP-only fallback, reconnect, resubscription, and hydration transitions when relevant.
strategies/AGENTS.md
# Strategy guidance

Strategies consume prepared context and return deterministic candidates or no-trade decisions.

They must not select contracts, call broker APIs, fetch instruments, or bypass readiness and risk layers.

Every mandatory input must be declared. Missing optional context may reduce confidence but must not become a hard blocker unless explicitly required by that strategy.

Any scoring change requires focused tests and evidence against overfitting. Do not infer profitability from in-sample results alone.
NIMS-Chrome/AGENTS.md integration section
## Integration with nifty_scalper_bot

Before changing a shared command, configuration key, message payload, authentication mechanism, API endpoint, or user-visible trading workflow:

1. identify the contract owner;
2. inspect the corresponding consumer or producer in `nifty_scalper_bot`;
3. preserve backward compatibility where feasible;
4. make coordinated changes only when required;
5. validate both repositories and document any staged rollout requirement.

Do not duplicate trading strategy, instrument selection, risk, or broker-execution logic in this repository.
4. Reusable Task Prompts
Routine fix or implementation
Implement the requested change in accordance with the active AGENTS.md files.

Outcome:
[Describe the behavior that must work.]

Observed problem or reproduction:
[Logs, error, steps, screenshots, or failing test.]

Scope:
[Relevant files/components if known.]

Cross-repository context:
[State whether NIMS-Chrome or another repository may share the affected contract.]

Constraints:
[Only task-specific boundaries not already present in AGENTS.md.]

Definition of done:
- reproduce or establish the current failure;
- identify the owning component and root cause;
- implement the smallest coherent fix;
- update dependent repositories only if the traced contract requires it;
- add focused regression coverage;
- run relevant validation and report actual results.

Proceed without pausing for routine in-scope local edits. Ask only if a missing decision materially changes behavior, safety, or scope.
Short task prompt
Follow the active AGENTS.md instructions.

Fix: [exact problem and expected behavior]
Evidence/repro: [logs or steps]
Scope: [known component]
Cross-repo: inspect NIMS-Chrome only if the shared contract may be affected.

Reproduce or trace the failure, implement the smallest coherent fix, add regression coverage, run relevant checks, and report actual results. Preserve unrelated behavior.
Deep audit without editing
Audit the following problem without modifying code:

[problem]

Trace the complete runtime path and rank findings as critical, high, medium, or low.

For each actionable finding provide:
- observed evidence;
- file and function;
- exact failure mechanism;
- runtime consequence;
- whether NIMS-Chrome or another integration is affected;
- smallest appropriate correction;
- focused validation needed.

Separate confirmed defects from risks, design concerns, and unverified hypotheses. Do not infer a root cause solely from one log message.
Implement selected audit findings
Implement confirmed findings [IDs].

Preserve all unrelated behavior. Make coordinated NIMS-Chrome changes only where the audited integration contract requires them. Add focused regression tests and run the relevant validation.
Strategy optimization
Evaluate and improve [strategy] without weakening execution or risk controls.

Goal:
Improve risk-adjusted out-of-sample performance, not raw in-sample profit or trade frequency.

Preserve:
- option-only execution;
- capital and daily-loss limits;
- existing execution safety;
- deterministic reproducibility.

Evaluation must include, where data permits:
- transaction costs, brokerage, taxes and realistic slippage;
- train/validation/test or walk-forward separation;
- market-regime breakdown;
- trade count and exposure;
- maximum drawdown;
- expectancy and profit factor;
- sensitivity to parameter changes;
- comparison against the unchanged baseline;
- detection of look-ahead, survivorship and selection bias.

Do not implement a parameter or logic change merely because it improves one backtest. Report whether the evidence supports implementation, experimentation, or rejection.
Recommended Placement
Content	Location
Medical, research, coding, trading and response preferences	Global custom instructions
Trading-bot architecture and invariants	nifty_scalper_bot/AGENTS.md
Market-data, execution and strategy rules	Nested AGENTS.md files
Browser-extension integration rules	NIMS-Chrome/AGENTS.md
Exact problem, evidence and definition of done	Individual task prompt
Mechanical command restrictions and approvals	Codex configuration/rules
Keep durable instructions concise. Add a new repository rule only after a repeated failure or when it protects an important invariant.