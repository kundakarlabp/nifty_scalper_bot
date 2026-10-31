# P4/P5 prospective evidence protocol — 31 October 2026

## Current conclusion
No P4 iron-condor or P5 long-straddle profitability has been demonstrated on
fresh prospective live or paper data. Code tests and the synthetic examples in
`tests/backtests/test_volatility_prospective_fills.py` are **not trading outcomes**.
Do not toggle LIVE, or promote strategies to the production catalog from a CI
pass or a backtest alone.

## Capture: every decision, including refusals
Run P4 and P5 in explicitly opted-in SHADOW observation during market hours.
For **every** eligible evaluation time and instrument basket, retain:

1. Strategy identifier; session date; decision/session timestamp; deterministic
   decision ID; selected expiry, strikes, instrument tokens and exchange symbols;
   broker-dump lot size and intended parent leg roles.
2. The actual per-leg synchronized FULL WebSocket bid/ask/depth and
   exchange timestamps, bid/ask quantities, spread, IV source, forecast
   timestamp and volatility regime. Stale/missing/crossed/partial depth means
   **abstain**. Store all no-trade gate reasons with the same rigor as trades.
3. Existing account- and risk-authoritative limits, orderbook, position
   snapshot, cash/margin quote time, risk and cost assumptions, entry/exit
   policy version, code SHA, mode, and restart/reconnect events.
4. **No synthetic fills presented as broker fills.** Label shadow
   mark-to-market, paper simulated execution and actual broker order fills
   separately. A hypothetical bid/ask at signal time is never an actual fill.
5. For every child order: stable parent/child identity, order request time,
   broker order ID, cumulative filled quantity, average executed price,
   orderbook statuses, partial fills, timeout/cancel/reject/modify, confirmed
   broker net-position snapshot, protections, force-exit and eventual flatten.
6. Preserve exit/settlement fee breakdown and observed slippage (including
   stressed tails), not simply gross expiry max-loss/payoff.

## Analysis: prespecify before observing outcomes
- Freeze entry/exit logic, risk limits, forecast calibration and reporting
  definitions. Keep the forward observation period untouched; document any
  material version change and restart its validation cohort.
- Chronologically replay earlier independent FULL-depth data with the same
  option contracts and execution cost model; stratify by low/high realized vol,
  IV/forecast gap, expiry distance and market regimes.
- For paper evidence, use the actual market-depth constraints, limited
  queue/fill assumptions, fees, liquidity, partial-fill failure scenarios and
  extreme bid/ask widening. Never mark all intended legs filled from a single
  snapshot; sequential Zerodha orders have execution gap risk.
- Report opportunity coverage including rejected/unfillable/missing-data
  opportunities. Include number of complete strategies, incomplete orders,
  loss events, signal availability, costs, all-in net expectancy per strategy
  and regime, win rate, payoff distribution, downside tail and peak-to-trough
  drawdown with uncertainty intervals; show sensitivity to stress assumptions.
- Use an immutable timestamped observation manifest and reconcile event counts
  to the market evaluation clock to detect missing signals, selective censoring,
  restarts, and survivorship/look-ahead bias.
- Compare P4 and P5 to a **no-trade** baseline, risk-matched existing strategy
  portfolio, and an IV/volatility-conditioned simple benchmark. Do not infer
  alpha from an unadjusted positive sample mean or from one market regime.
- Run controlled broker-free recovery drills for partial wing, partial
  short fill, lost acknowledgement, cancel after fill, restart during order
  placement, expired option, rejected exit, and corrupt state.

## Typed after-cost evidence utility
`backtesting.volatility_prospective.reconcile_prospective_trade` reconciles
complete paired entry and exit fills for all intended legs and estimates each
order's charges using the existing canonical cost model. It requires fresh
FULL-source decision quotes and explicitly labeled `BROKER` versus `PAPER`
fills. A missing or invalid order raises rather than estimating profit.
It is **not** the acquisition workflow and does not establish comprehensive
opportunity coverage, bid/ask historical completeness, real exchange fees,
tax settlement liability or risk-adjusted alpha.

## LIVE go/no-go
**HOLD** until: credible chronological out-of-sample plus prospectively
captured SHADOW/PAPER evidence; signed-off margin and portfolio-level net loss
bounds; durable parent reservation; canonical owner-controlled wing-first
entry, partial fill and timeout recovery; broker-confirmed exposure and
coordinated group stop/EOD flatten; emergency operator rollback drills;
real-mode read-only evidence and explicit human approval. No automatic LIVE
activation, no real-money test order, no unsupported profitability claim.

Tracking: issue #1951. Keep the normal single-leg strategy pipeline unchanged.
