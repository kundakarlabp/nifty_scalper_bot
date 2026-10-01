# Strategy and execution follow-up — 1 October 2026

## Observed production state

Read-only relay observations at approximately 15:18–15:28 IST showed build
`518a4e4c5dc15f6d6e0d13983b537a75758d8efa` alive, authenticated, reconciled and
armed. Both selected options had adequate history. Recent runner summaries
showed zero candidates; capital/readiness did not explain that interval.
The recent ERROR-filtered log request returned no matching lines.

| Decision | Owner and interpretation |
| --- | --- |
| `single_vote_scalp_disabled` | StrategyManager explicitly disables unconfirmed lone triggers by default. This is configuration policy, not proof of a runtime defect. |
| `single_trigger_context_confirmation_invalid` | Fresh, strong, eligible OrderFlow context is required for the confirmed single-trigger path. Missing eligibility must not be invented. |
| `underlying_direction_conflict` | Canonical underlying direction conflicts with the candidate option side. A falling day alone does not establish the correct direction at a particular entry time. |
| `alpha_below_threshold` | Runner observed final score 8.47 but independent alpha 6.33 against 7.50 for VWAPPro. Execution quality cannot manufacture directional edge. |

Health readiness means permission to consider an order; it does not promise a
strategy candidate or a broker fill. These logs are a short recent interval,
not a complete explanation of the entire trading day. No captured live input
replay established that the timestamp defects below caused these live rejections.

## Reproduced defects and owner-consistent corrections

1. StrategyManager used `vote_timestamp` for context freshness but separately
   used `timestamp_epoch`/`ts`/default-now for hard veto age. Conflicting metadata
   bypassed a fresh veto or imposed an expired one; malformed legacy values
   raised. Future/infinite vote timestamps also passed freshness. Reuse the
   existing freshness authority for confirmation and veto, with the tighter
   configured veto-age cap. Preserve soft penalties and valid fresh vetoes.
2. OrderManager modification called a nonexistent `get_order` method. Four
   repricing/resize call sites also used unsupported `new_price`/`new_quantity`
   keywords. Use the existing locked order store and existing public keywords.
   Omitted fields are now omitted from broker modifications instead of sent as
   zero prices/triggers/quantity. Empty modifications and unknown orders fail.
3. Partially filled stop/target resizing sent outstanding quantity to the broker
   but recorded filled-plus-outstanding locally. Compute the total once and
   use it for stop/primary/secondary broker modifications and local state.
4. The required agent validation façade called `_run` with an unsupported
   `cwd` keyword. Pass its existing repository-root argument; a regression
   proves compile failures still stop validation before style mutation.

These changes add no alternate strategy, execution route, monkey patch,
dependency, manager, risk override or live threshold adjustment.

## Research and established system practices

| Primary source | Useful finding | Application and limitation |
| --- | --- | --- |
| [Cont, Kukanov & Stoikov: The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402) | Short-horizon price changes relate to order-flow imbalance and market depth in 50 US stocks. | Study temporal OFI rather than treating static depth as independent proof of alpha. Transfer to NIFTY option premiums requires local data and holdout testing. |
| [Gao et al.: Market intraday momentum](https://www.sciencedirect.com/science/article/pii/S0304405X18301351) | First-half-hour S&P 500 ETF return predicts the last half-hour in the studied period. | Candidate for a session-specific underlying-context study; not evidence for arbitrary minute scalps or long-option profitability. |
| [Zarattini, Barbon & Aziz: ORB working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284) | Examines 5-minute opening-range breakouts in US equities. | Working-paper evidence and a different universe/instrument. Do not copy parameters or claim validated NIFTY option alpha. Full text was not retrieved in this audit. |
| [Bailey et al.: Probability of backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | Repeated strategy selection can overfit backtests. | Predeclare one hypothesis, record variants, use chronological holdouts and parameter stability. |
| [QuantConnect LEAN fill models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts) and [slippage](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts) | Fills model execution price/quantity; spread and slippage affect realized results. | Extend existing replay/fill seams with observed latency and spread distributions rather than installing a second trading framework. |
| [NautilusTrader reconciliation](https://nautilustrader.io/docs/latest/concepts/execution/reconciliation/) | Venue reports reconcile order/fill/position state. | Preserve current broker-state ownership, idempotency and recovery. These are operational practices, not profitability evidence. |
| [Kite order API](https://kite.trade/docs/connect/v3/orders/) | An order ID acknowledges placement; it does not prove execution. Modification exposes independent quantity, price and trigger fields. | Preserve broker-confirmed fills and modify only intended fields; verify protective exit acknowledgements separately. |

## Next research experiment within existing architecture

Use existing replay/parity and completed-trade analysis modules to compare the
unchanged baseline against one prespecified OFI or session-context hypothesis.
Include unsuccessful candidates, rejected/partial orders, actual bid/ask,
latency, brokerage/statutory charges and slippage. Split chronologically by
session and expiry, keeping a final untouched holdout. Report net expectancy,
trade count, drawdown, tail losses, turnover, regime sensitivity and parameter
stability. Require enough data for uncertainty estimates before changing live
admission thresholds. This audit does not establish a profitable strategy.

## Validation scope

Tests first reproduced timestamp/veto, façade invocation, public modification
and partial-target quantity failures. Corrections pass their focused regressions.
Run `scripts/agent_task.py full` for the changed files and final-head GitHub CI
before merge. Broker-free simulation verifies engineering contracts, not live
order execution or trading expectancy. Production was not restarted or redeployed.

## Signal evidence follow-up (issue #1468)

Objective: correct the existing temporal OFI authority and close missing research
coverage without adding a strategy, scoring owner, broker path or dependency.

Deterministic tests reproduced cached one-second OFI remaining ready on duplicate
versions after its events expired, infinite ask/queue quantities passing book
validation, and a bounded event buffer reporting truncated flow as ready.
The accumulator now ages windows on every call without counting duplicates,
rejects non-finite books/clocks, preserves the accepted book on late updates,
and marks truncated windows explicitly. One-second readiness recovers when
discarded events leave that window; three-second completeness remains separate.
The unused cached snapshot was removed. Runner forwards both completeness facts.

| Review recommendation | Canonical implementation and disposition |
| --- | --- |
| Calibrate score/confidence | Reuse `calibrate_signal_scores` and the existing report's alpha/final/setup bins, net expectancy and bootstrap uncertainty. A heuristic confidence is not a win probability. No new calibration layer or unsupported threshold change. |
| Target reachability | Extend existing `label_forward_option_buy_path` with optional target and first target/stop timestamp. Original and adjusted targets can be compared on the same executable bid path. Same-timestamp contradictory barriers are explicitly ambiguous. MFE/terminal return remain full-horizon descriptive fields, not simulated realized returns. |
| Correlated features | Reuse independent setup-score lineage and aligned candidate-return PBO/DSR analysis. Feature removal requires a predeclared baseline-vs-ablation replay, not a claim that correlated names prove redundancy. |
| OFI reliability | Apply the reproduced freshness, finiteness, late-update and bounded-window corrections in the existing accumulator; retain existing context role and formula. |
| Session/expiry behaviour | Extend the existing outcome-cohort function and report with IST decision hour, calendar days to contract expiry, and target-adjustment status. Missing, future or invalid timing facts stay `unknown`; no close-hour proxy or inferred expiry. |
| Duplicate gates | Reuse deduplicated setup opportunities, labelled gate-effectiveness and replay parity. No newly demonstrated duplicate admission rule justified removing safeguards. |
| Executable costs/fills | Reuse the canonical fee model, broker-cost ledger, execution-quality provenance and bid-path labeller. Runner now preserves existing expiry/original/adjusted-target facts in trade provenance. No replacement fill model. |
| Holdouts/ablation | Reuse experiment records, chronological walk-forward, aligned candidate returns and PBO/DSR. Retain a final untouched session/expiry holdout and fit any calibration only on preceding data. |

Run the existing report with a locally authorized canonical journal:

```bash
python scripts/reporting/analyze_completed_trades.py --trades-db /path/to/trades.db
```

The report adds `by_entry_hour_ist`, `by_days_to_expiry` and
`by_target_adjustment` to `realized_post_cost_evidence`. These descriptive cohorts
must be assessed within strategy/setup/regime and with sample size uncertainty;
pooled differences do not establish a causal effect. Legacy rows remain included
under `unknown` when new provenance is absent. Gate analysis additionally needs
independent opportunity labels; ablation/selection analysis needs aligned
post-cost candidate returns. Do not silently substitute completed winners for
the full opportunity population.

Research basis remains the primary sources above: Cont et al. support studying
temporal OFI/depth, Gao et al. support horizon-specific session hypotheses, and
Bailey et al. support controlling strategy-selection bias. LEAN/Nautilus provide
execution/research engineering practices, not proof of NIFTY option profitability.
The canonical fee defaults were checked against
[Zerodha's current charges](https://zerodha.com/charges/).

No adequate chronological NIFTY option dataset or independently labelled
counterfactual sample was supplied for this follow-up. Synthetic regressions
validate engineering behaviour only; they cannot justify new live score weights,
session filters, target multipliers or removal of valid gates. Runtime admission,
risk limits and target-repair policy remain unchanged. OFI corrections can change
confirmation when the old evidence was stale, invalid or truncated. Rollback is
a revert of the merge; additive outcome/report fields require no schema migration.
