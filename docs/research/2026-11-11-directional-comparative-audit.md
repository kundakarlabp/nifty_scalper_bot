# 11 November 2026 — directional strategy research protocol

## Decision to test

**Do not infer alpha from rejection frequency.** The 11 November production archive
contains 150 strategy status windows, 84 with a nonzero strategy evaluation,
zero generated trade candidates, and repeated VWAP `distance_outside_band`,
SMC sweep/confirmation, and ORB `orb_entry_window_expired` reasons. These
are observed vetoes, not known missed profits. Some status windows have zero
evaluations because the global engine/capital gate does not admit a candidate;
use the full gate funnel, not only the strategy denominator.

Retain the current LIVE SMC/VWAP/ORB policy and risk controls. Compare three
separate hypotheses through the existing SHADOW research simulators:

1. **VWAP continuation:** The already-implemented `strict_continuation_either_proof`
   versus `strict_continuation_retest_rebreak_only` pair tests whether a
   confirmed momentum breakout without retest adds net value. Same archive,
   option-side costs and 10/25/50-bps slippage; report suppressed signals and
   exit-resolution coverage. This is an *eligibility* ablation, not a changed
   live VWAP band.
2. **ORB entry timing:** Existing 120-minute baseline after the opening
   range, versus **30 / 60 / 180 minutes**, each with identical remaining
   controls and 10/25/50-bps slippage. The 180-minute arm is a *research-only*
   extension, not an instruction to extend late-day trading. Matched entry
   identities and changed trade cohorts are recorded separately; no
   counterfactual fill is created when baseline would not have traded.
3. **Independent failed-breakout reversal vs SMC:** Use the existing
   `failed_breakout_reversal` *two separately completed futures bars*
   hypothesis. Its immutable preceding 20-bar reference, 14 prior completed
   ATR ranges, sweep and confirmation-volume gates remain unchanged.
   Run real SMCStrategy and the research-only reversal adapter through the
   **same canonical `_scenario`** with identical historical option symbol,
   next-minute entry, 1.5 minimum net reward/risk, price/cost assumptions
   and 10/25/50-bps slippage. Match outcomes by the confirmed underlying
   event, option contract and next eligible entry minute. Do not add the
   research adapter to the LIVE builder.

## Reproducible research command

Install the repository in the research environment and provide a validated
archive directory in the existing `load_archive` candle format. Run in an
isolated SHADOW process, with no broker credentials or live order route:

```bash
EXECUTION_MODE=SHADOW python -m scripts.audit_directional_strategies \
  --archive /path/to/validated-research-session-archive \
  --output /path/to/new-directional-audit.json
```

Output reports each study, source SHA-256, eligibility/exit coverage,
matched-event fills, unresolved exits, and unpromotable research status.
The output file must not exist already. Do not execute this in the live
market process. The command never changes active settings or opens orders.

## Limits of the comparison

- Even the same futures event can have **different SMC and reversal signal
  times**. A same-minute next-open comparison measures contemporaneous
  admission, not complete attribution of an SMC sweep sequence.
- A missing trade is **not** proof the strategy voted no; it can reflect
  no signal, another open simulated position, missing next-bar liquidity,
  cost/geometry rejection or pending-intent expiry.
- The archived weekly-ATM filter is a historical *proximity proxy*. Current
  instrument masters must not be used to backfill past selection.
- Archived OHLC lacks point-in-time full broker bid/ask, depth, order queue and
  execution latency. Modeled canonical fees and 10/25/50-bps adverse slippage
  do not establish executable backtest parity.
- The last 20% of already inspected research sessions is a **retrospective
  check**, not a new untouched holdout. Evaluate future chronological sessions
  independently before selecting thresholds.
- A winning reversal-only proxy trade or larger aggregate P&L is not sufficient
  to alter the LIVE SMC/ORB/VWAP strategy. Require matched opportunity
  denominators, samples across regimes, drawdown/tail risk, costs, unresolved
  exit accounting, forward SHADOW and paper broker-safe evidence.

## Readiness for policy decisions

Keep **`promotion_eligible=false`** until the independent holdout and
prospective results show a robust after-cost benefit with realistic strikes,
quotes, fills, and no change to daily risk, broker safety or stops.

If source coverage is insufficient or a strategy has no fills, the correct
verdict is **INCONCLUSIVE / NO EVIDENCE**, not a profitable or unprofitable
strategy classification. Preserve all attempted variants to avoid selection
bias and continue collecting true historical option observations.
