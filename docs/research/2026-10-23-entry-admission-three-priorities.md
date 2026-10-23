# October 23: narrowly bounded entry-admission hypotheses

## Scope and attribution
Observed production: the engine was healthy and armed, but most evaluations
produced no trigger. Repeated OrderFlow observations are not distinct
opportunities; distinct SMC setup IDs and option quote timestamps must be
deduplicated. A post-trade profitable counterfactual has **not** been proved.

## Single owners and experiment switches

| Question | Canonical owner | Opt-in flag (default false) | Unchanged safety gates |
| --- | --- | --- | --- |
| Qualified SMC with two separately timed supportive books | `SustainedSMCFlow` called by `StrategyManager` | `SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED` | SMC structural setup ID, direction, opposing-context veto, capacity-retry consensus, broker/runner/risk |
| OFI reversal persistence versus latency | `OrderFlowStrategy` | `ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED` | Base holdoff 15s; minimum 10s when opted in and 3s sustained OFI and strong same-side book support; adverse veto |
| Mid-trend premium pullback and rebreak beyond ordinary ATR VWAP band | `VWAPProStrategy` | `VWAP_TREND_RETEST_EXTENSION_ENABLED` | Session VWAP, strong fresh underlying direction, TREND regime, finalized current and previous adjacent bars, bearish prior retracement, rebreak, option RVOL, futures slope and risk |

No alternate entry arbiter, second quote freshness authority, second stop/sizing
system, or unbounded global VWAP distance increase is introduced.
The SMC proof engine may observe evidence with its admission flag off:
`SMC_SUSTAINED_FLOW_PROOF` reports `is_counterfactual=true` and
**does not authorize a live order**. Proof is consumed and logged only once
per unique setup identity; subsequent quote versions or adverse transitions
cannot manufacture additional missed opportunities.

The OrderFlow knob `ORDERFLOW_OFI_REVERSAL_HOLDOFF_SECONDS` retains its
existing default of 15 seconds. Adaptive adjustment is limited to 10 seconds
only after **complete** and **sustained** supportive 3s OFI plus strong depth.
The experimental path enforces a hard 10-second minimum even if the legacy
holdoff environment setting is shorter; with the experiment off, explicit
legacy settings remain authoritative.
Missing, neutral, stale or adverse flow retains the original holdoff.

VWAPPro's `VWAP_TREND_RETEST_EXTENSION_MAX_ATR` defaults to 3.5;
the opt-in alternative path still uses the existing session- and
option-contract-specific thesis and setup ID, preserving stop rearm semantics.

## Required promotion evidence
1. Pass complete exact-head GitHub CI, safety and temporal regression tests.
2. Replay baseline and one-switch-at-a-time ablations on capture-complete
   matching option contracts and underlying futures, with unique setup IDs.
3. Verify executable bid/ask at signal timestamps, slippage, all costs,
   stop/fill behavior, capital/risk affordability and unresolved outcomes.
4. Compare candidate counts, rejection stages, net expectancy, drawdown and
   false-positive rate over development, separate validation and current
   weekly-options out-of-sample sessions. No profitable-miss inference from
   no-trade counters alone.
5. Only with convincing evidence and explicit operational approval, activate
   one independent flag; otherwise leave all three disabled.

Deployment during the live session must not interrupt a broker order or
position. A merge alone is not proof production has deployed that SHA.
