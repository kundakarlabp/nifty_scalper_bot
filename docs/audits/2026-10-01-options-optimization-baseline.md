# Options optimisation: research baseline and first corrections

## Expert judgement

The bot has reusable strategy, market-data, risk, execution and research owners.
The highest-value next step is establishing dependable post-cost evidence for
those owners, not adding ten triggers or fitting more thresholds to a few trades.
Correctness, plausible market mechanisms and measured profitability are distinct.
This work establishes engineering correctness only; it does not establish alpha.

Baseline: main `f0a43aa005543d82cb7ae8b3b04bd265a4c3f4e5`.
Scope: issue #1470, backtest initial-capital accounting and replay chronology.

## Available data

The workspace contains `sample_replay.csv` (6 observations), `sample_ticks.csv`
(3 observations), and the canonical golden replay (3 observations). These are
engineering fixtures, not a multi-session NIFTY options research sample.
The two local state databases contain no order, position, lifecycle-event or
bracket rows. No canonical completed-trade journal or historical depth archive
was available in this workspace. No live broker request, order, restart or
deployment was performed to obtain data.

Kite's historical API supplies OHLC/volume/OI candles, not historical bid/ask
depth. Its instrument master only returns live contracts; expired token maps
must have been cached, and continuous history is a futures/day-candle facility.
Do not manufacture option depth, spreads, historical ATM baskets or missing OFI
from candles. Current instrument selection is not a historical universe.

## Reproduced defects and minimal corrections

| Defect | Existing owner | Correction |
| --- | --- | --- |
| Total return starts at first post-entry equity, omitting initial costs | `backtesting/backtest_engine.py` | Use configured initial capital; include first observation's account return |
| Drawdown ignores the initial capital peak | Same owner | Bound running peak below by initial capital |
| Exported total/daily P&L omits entry costs and overnight changes | Same owner | Compute total from initial capital and daily changes from preceding observed day-end equity |
| Direct dataframe replay runs input order while file replay sorts | `backtesting/replay.py` | Own stable chronological ordering once in `run_dataframe`; remove duplicate loader conversion/sorting |
| Missing timestamps reach clock/strategy processing | Both input owners | Reject missing timestamps before dispatch or strategy evaluation |

No new manager, strategy, scoring authority, execution path or dependency was
introduced. Strategy weights, expiry activation, risk limits, stops and targets
were not tuned on the engineering fixtures.

## Baseline comparison actually run

These deterministic accounting experiments use initial capital 10,000, quantity
10, zero slippage and **1% commission deliberately chosen for visible arithmetic**.
This is not the bot's live fee configuration or a market profitability backtest.
The test also covers an entry with both commission and slippage.

| Fixture | Final equity, unchanged | Old return | Correct return | Old max drawdown | Correct max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entry only at 100 | 9,990 | 0% | -0.1000% | 0% | -0.1000% |
| Flat round trip at 100 | 9,980 | -0.1001% | -0.2000% | -0.1001% | -0.2000% |
| Entry 100, exit 120 | 10,178 | +1.8819% | +1.7800% | 0% | -0.1000% |
| No trade | 10,000 | 0% | 0% | 0% | 0% |

Orders, fees, final equity and closed-trade P&L are unchanged for these cases.
Reported returns now compound to final equity relative to initial capital.
Additional replay tests compare unsorted dataframe/file input, preserve equal
timestamp record order, preserve caller input, and reject missing timestamps
before any tick dispatch or clock advance. Six regressions failed before the
implementation; the initial focused run passed 23 tests before adding the extra
commission-plus-slippage case.

Final review reproduced five export-accounting failures. The JSON total now
agrees with account equity and its daily values telescope to that total, including
initial costs and overnight movement. A multi-session fixture with a weekend gap
reports daily P&L of -11, +100 and -61.5, summing to +27.5 without inventing weekend
observations. This is research accounting, not a change to live overnight policy.

## Prioritised experiments using existing code

Each row is a separate hypothesis, not a proposed unconditional live change.

| Priority | Area | Prespecified comparison | Existing implementation/research seam |
| --- | --- | --- | --- |
| 1 | ORB | Fresh breakout versus confirmed retest within the current session window; volume-normalised cohorts | `orb_pro.py`, setup opportunities, replay |
| 2 | VWAP | Native premium reclaim strength versus overlapping directional/context contributions | `vwap_pro.py`, independent setup lineage, score calibration |
| 3 | Squeeze | Compression-plus-expansion versus existing premium momentum; ablate duplicated VWAP/RSI contributions | Runner's existing squeeze builder, BB context, canonical score policy |
| 4 | Reversal | Completed-bar sweep/reclaim versus persistent trend; invalidation and same-setup retries | `smc_liquidity.py`, setup lifecycle, setup-opportunity deduplication |
| 5 | OFI | Base trigger versus fresh/complete OFI confirmation | Existing accumulator and OrderFlow context; no separate trigger |
| 6 | Stops/targets | Original versus cost-repaired target on identical entry opportunities; compare first stop/target, not full-horizon MFE as realised P&L | `premium_risk_geometry.py`, bid-path labels, original/adjusted target provenance |
| 7 | Exit timing | Current exit versus bounded no-progress/time exit while preserving hard loss cap | Existing BracketManager; candidate only after full quote-path capture |
| 8 | Contract choice | Current ATM policy versus neighbouring eligible strike, stratified by delta, spread and actual DTE | InstrumentManager historical snapshots; no strategy-local selector |
| 9 | Execution | Observed spread/latency/depth versus fill assumptions; include rejected and partial orders | Existing PaperFillEngine calibration, broker ledger, execution provenance |
| 10 | Gates | One labelled opportunity per setup; assess rejected opportunities alongside accepted ones | Existing candidate journal, gate analysis, setup opportunities |

The squeeze builder currently uses premium VWAP/EMA/RSI momentum; its name alone
does not demonstrate compression. It also awards related premium conditions to
both direction and setup scores. Runner's downstream direction/context authority
must be traced before interpreting or changing those contributions. Do not label
all such contributions a proven defect or remove them without an ablation.

The current cost-floor helper repairs targets without widening stops. Higher
arithmetic R:R does not imply a more reachable target or improved expectancy.
Evaluate original and adjusted targets on the same executable path, accounting
for gaps/slippage at stops and actual exit-order semantics.

## Research acceptance and limits

Use the existing experiment record, canonical completed-trade analysis, score
calibration, chronological evaluation, PBO and deflated-Sharpe tools. Candidate
and baseline returns must refer to aligned opportunities. Completed trades alone
cannot establish what would have happened to rejected opportunities.

Split by complete sessions/expiries, purge overlapping labels, fit only on prior
history, and reserve an untouched final holdout. Trade-count expanding windows
are descriptive diagnostics; they are not automatically a session-separated,
purged parameter-selection experiment. Include all tried variants, not only the
winner, and prefer a stable parameter region to one isolated optimum.

Report net expectancy and uncertainty, total net P&L, drawdown, tail loss, profit
factor, trade count, turnover, exposure, fill/rejection rate and session/expiry
sensitivity. Assess correlated intraday observations with session-aware
uncertainty; the existing IID bootstrap alone does not establish independence.

The vector engine still uses a generic proportional commission and configured
price column; it is not the canonical NIFTY fee/depth execution model. Its
annualised metrics now use observed daily closing equity with 252 sessions/year
(see the subsequent [session-metrics correction](2026-10-01-backtest-session-metrics.md)).
Short or incomplete histories still do not establish reliable annual performance.
It also passes the whole input frame to a strategy;
feature causality is the strategy's responsibility. Bar-start versus actual
availability timestamps must be explicit before using candle OHLC for decisions.
Minute-candle extrema cannot establish tick-level target/stop ordering or OFI.
These limitations prevent using the fixture results as profitability evidence.

## Evidence and system references

- [ORB working paper, university full text](https://www.alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content): US equity selection/activity evidence; not a NIFTY options replication.
- [VWAP study, author publication](https://concretumgroup.com/volume-weighted-average-price-vwap-the-holy-grail-for-day-trading-systems/): US ETF backtest; not proof for exact reclaim rules or long-option economics.
- [Cont, Kukanov and Stoikov](https://arxiv.org/abs/1011.6402): contemporaneous OFI/depth price-impact relationship; not post-latency trading alpha.
- [Probability of backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf): selection bias requires explicit control.
- [LEAN fills](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts): execution price/quantity modelling rather than candle-price optimism.
- [Kite historical candles](https://kite.trade/docs/connect/v3/historical/) and [FULL streaming](https://kite.trade/docs/connect/v3/websocket/): distinct historical-candle and live-depth evidence.

Rollback: revert the focused merge. No migration is needed. There is no supported
claim of profitable strategy tuning, historical NIFTY holdout support or deployment.
