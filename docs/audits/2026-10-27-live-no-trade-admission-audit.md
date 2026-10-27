# 2026-10-27 LIVE no-trade admission audit (read-only)

## Evidence and running version

- Production Supabase project: `dehdptgkqbrkyzyodicd`.
- Inspected `nifty_runtime_heartbeat`, `nifty_daily_market_logs`, and
  `nifty_trade_events`; log source is Lightsail journald.
- LIVE build `548d12204cec5e9ab0c7ed589a09611ea1ae26a7`,
  trading status `armed`, engine healthy, no top-level blockers, no entry
  events recorded in the inspected interval.
- Production startup reported
  `SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=True mode=LIVE`,
  `LIVE_ENTRY_QUALITY_CONFIG vwap_retest=True adaptive_ofi=True mode=LIVE`,
  and `LIVE_RISK_ENVELOPE_CONFIG per_trade_pct=5.0 daily_loss_pct=5.0`.
  The 5% values are reported settings, not risk-change recommendations.

## Setup identity and admission reconstruction

One SMC CE structural setup `smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-27T05:59:00+00:00`
received repeated trigger votes around 11:34–11:35 IST; these are **not**
independent trading opportunities.

| IST | Evidence | Counterfactual proof implication |
| --- | --- | --- |
| 11:34:05 | SMC `TRIGGER_QUALIFIED`; temporal OFI aligned, fresh | First admissible book observation only; entry rejected `smc_independent_confirmation_missing` |
| 11:34:10 | Fresh adverse microstructure; opposing context | No second proof; entry withheld |
| 11:34:15 | Further adverse microstructure | No sustained confirmation |
| 11:34:21–11:34:36 | Holdoff or non-confirming OFI | No qualifying consecutive pair |
| 11:34:42 | Three-second OFI not aligned | No qualifying observation |
| 11:34:47 | Temporal OFI aligned, fresh | New first observation |
| 11:34:52 | Three-second OFI not aligned | Interrupts proof sequence |
| 11:34:57 | Temporal OFI aligned, fresh | New first observation |
| 11:35:00 | Independent flow confirmation missing | No second proof |

The repository's `SustainedSMCFlow.observe` requires two different quote
versions with complete sustained book support 4–12 seconds apart; an
unqualified intervening observation resets the unconsumed proof.
No `SMC_SUSTAINED_FLOW_PROOF` event appeared in the archived session logs.

**Inference:** the recorded order-flow sequence does not justify bypassing
the independent SMC confirmation. A single-book entry at 11:34:05 might have
created an order, but this is not evidence of a profitable missed trade.

## Directionless evaluations

Around 11:30–11:31 and 11:46–11:47 IST, the runner logged
`direction_context_unresolved_live` with
`spot_fresh=True fut_fresh=True` on selected CE and PE.

This is **not** proof of a code bug or stale market data: freshness is not
direction validity. The runner requires a fresh, resolved underlying CE/PE
bias and intentionally fails closed. Both the runner and StrategyManager
contain logic to reject conflicting underlying directions.

The original aggregated log line exposed only freshness, without the
underlying spot/futures biases and their direction-reason details. The
accompanying code-only diagnostic patch logs this missing attribution,
without changing signal admission, capital, stops or order execution.

## Operational observations

- Runner restarted at 11:10 and 11:17 IST, then returned to LIVE_READY.
  Cause is unverified; do not assume restart caused the absence of entries.
- Monitoring relay returned 404/502 in prior probes; Supabase heartbeat and
  archive remain available. A failed relay is not engine-down evidence.
- Frequent `same_bar_periodic_eval_throttled` entries represent the
  intentional periodic gate, which is tested to permit fresh-bar evaluation
  and retry after underlying context generation changes.
- VWAP, SMC and ORB non-votes dominate the observed session.
  OrderFlow is context-only by design; it cannot independently trigger a trade.

## Validation boundary and next experiments

A full economic counterfactual requires recorded completed spot/futures
candles, exact option contract identifiers, minute- and quote-level bid/ask,
timestamped depth, broker executable fills, spreads and transaction costs.
The inspected 11:34 journal window contains decisions and OFI evidence,
**not** executable bid/ask or premium-price snapshots; it cannot establish
net P&L, adverse excursion or expected profit.

Next research must deduplicate setup IDs, replay each proposed gate
modification independently on capture-complete development and held-out
sessions, and compare approved entries, realized net P&L, stops, slippage,
false positives, drawdown and capital usage. Do not weaken directional,
stale-data, risk or adverse-flow protections merely to increase trade count.

This PR changes *diagnostic logging only*. It is safe to review while the
engine runs, but should not be merged/deployed during an open position.
