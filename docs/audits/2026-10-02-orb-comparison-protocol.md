# ORB bounded comparison protocol — 2 October 2026

## Objective and boundary

Add a reproducible ORB candidate comparison to the existing read-only, SHADOW-only archived-history worker. No live strategy defaults, protective stops, quantity/risk limits or order interfaces change. Existing AWS release polling launches immutable request `research-20261002-05`; dashboard requests use the same worker and expose `orb_comparison.json`.

The archive has already been inspected. Its last 20% of option sessions is a **retrospective chronological check**, not an untouched out-of-sample test. Do not use that check for candidate selection or claim validated alpha. Active contracts were selected retrospectively; the basket does not reconstruct historical ATM/expiry rotation. Component research is not full live-pipeline parity.

## Registered candidates

Compare the raw component reference, a cost-feasibility reference, and seven one-hypothesis candidates relative to the cost reference:

| Candidate | Change | Hypothesis |
|---|---|---|
| Raw reference | Existing production component settings | Preserve previous research baseline |
| Cost-gated reference | Net reward/risk >=1.5 at next-minute slipped entry | Avoid trades whose bracket economics cannot pay assumed costs |
| Retest only | ORB_MOMENTUM_BRANCH_ENABLED=false | Confirmation may reduce false breakouts |
| Earlier entry | ORB_MAX_ENTRY_MINUTES_AFTER_RANGE=60 | Late breaks may have less remaining follow-through |
| Wider target | ORB_TARGET_RR=2.2 | Larger winners may offset fixed order costs; hit rate may fall |
| 5-minute range | orb_minutes=5 | Faster price discovery, potentially more noise |
| 10-minute range | orb_minutes=10 | Intermediate range-length sensitivity |
| 30-minute range | orb_minutes=30 | More confirmation, potentially fewer opportunities |
| Opening RVOL | Futures opening volume / previous 14 complete same-window sessions >=1 | Avoid below-normal opening participation |

No Cartesian parameter search or post-result combination. All runs use one archived lot, canonical estimated fees and 10/25/50 bps adverse slippage per side. The cost gate uses slipped stop/target exit prices, does not repair targets or widen stops, and is a research-only approximation rather than the live cost-repair/arbitration pipeline. Both references include fees in realized outcomes; raw does not mean pre-cost.

A 5/10-minute range still respects earliest signal bar start 09:30 IST and next-minute execution. It is not a replication of papers entering immediately after a five-minute range. Opening RVOL is unavailable unless all 14 preceding session windows and the current window are complete; no future session or full-day current volume is used.

## Selection and required evidence

Rank candidates with development trades in all three slippage scenarios by their worst development-period net expectancy. Exclude the retrospective final-period check from ranking. Zero trades mean abstention, not alpha. Report all candidates, costs, trade counts, profit factors, average wins/losses, realized drawdown, exposure, turnover, no-vote and exit reasons. Thirty development trades is only a preliminary floor for further study, not a sufficient statistical validation threshold.

No candidate is automatically eligible for live promotion: missing historical contract/quote fidelity, reused data, prospective chronological validation and paper observation remain explicit blockers. A best observed research candidate is not a best future strategy.

## Literature and transfer limits

- Zarattini, Barbon and Aziz (2024), *A Profitable Day Trading Strategy for the U.S. Equity Market*: 7,000 US stocks, 2016–2023, opening relative volume versus 14 prior same-window volumes and a cross-sectional top-20 stocks-in-play portfolio. Daily ATR stops and end-of-day exits differ from NIFTY option-premium stops/targets. Commission assumptions do not establish executable NIFTY depth/slippage. https://www.alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content
- Holmberg, Lönnbark and Lundström (2013), *Assessing the profitability of intraday opening range breakout strategies*, Finance Research Letters 10:27–33: volatility thresholds on crude-oil futures, bootstrap significance. Supports a testable breakout mechanism, not a universal range length or NIFTY option profitability. Full-text retrieval unavailable; university author abstract reviewed. https://www.usbe.umu.se/enheter/econ/ues/ues845/index.html
- Lundström (working paper 2013, revised 2017), *Day trading returns across volatility states*: crude-oil and S&P 500 futures results depend on volatility state. Supports future causal regime stratification, not an ex-post high-volatility filter. Full-text retrieval unavailable; university author abstract reviewed. https://www.usbe.umu.se/enheter/econ/ues/ues861/index.html
- Bailey, Borwein, López de Prado and Zhu, *The Probability of Backtest Overfitting*: ordinary holdout checks fail to account for trial multiplicity and small samples. This experiment records all nine candidates/27 scenarios and makes no PBO or deflated-Sharpe estimate from inadequate data. https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- Zerodha current charges and April 2026 STT revision: options premium sell-side STT 0.15%, brokerage and exchange/GST/stamp costs matter. Canonical cost model already uses the current STT default; do not lower it to improve research performance. Actual operator overrides are recorded by the baseline report. https://zerodha.com/charges ; https://zerodha.com/marketintel/bulletin/445377/revision-in-stt-securities-transaction-tax-from-1st-april-2026

## Engineering validation

Red-capable tests cover environment restoration, bounded candidate count, explicit non-promotion, actual slipped-fill fee feasibility and prior-session-only complete-window RVOL. Existing tests cover causal next-minute entry, stop-first ambiguous candles, missing-minute cancellation, malformed/expired archives and real production component determinism. Run the public agent check/full facade and exact final-head CI before merge. These checks establish implementation behavior, not investment performance.
