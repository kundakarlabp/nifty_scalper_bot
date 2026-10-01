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
3. Partially filled target resizing sent outstanding quantity to the broker
   but recorded filled-plus-outstanding locally. Compute the total once and
   use it for both primary/secondary broker modifications and local state.
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
