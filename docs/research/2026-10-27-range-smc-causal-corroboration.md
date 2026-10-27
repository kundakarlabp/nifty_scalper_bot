# SMC RANGE corroboration: causal-domain safety correction

## Why this differs from threshold optimization

October 26, 2026, second trade: RANGE SMC CE entered after a roughly
6.5-minute-old futures sweep (already addressed by PR #1865).
The recorded only local setup support was option-premium BOS, without
retest, premium reclaim or underlying sweep volume expansion. Two timely
OrderFlow readings admitted it through the sustained-flow path.

The manager's BOS is computed using option-premium OHLC and OrderFlow
uses the same contract's book. Neither is an independent underlying
futures confirmation. Treating a single BOS as enough for RANGE SMC
therefore double-counted correlated evidence.

## Narrow production change

- RANGE SMC requires either **underlying sweep volume expansion**
  (the existing event volume threshold), or a **causally timestamped
  post-sweep retest and premium reclaim**.
- A lone option BOS/CHOCH is still recorded as diagnostic evidence, but
  it cannot by itself authorize RANGE SMC. It produces the explicit
  `smc_range_correlated_structure_only` no-vote reason.
- TREND SMC retains its current structural corroboration rule.
- The existing StrategyManager confirmation, selected contract, quote,
  age, affordability, risk, exchange stop, and broker guards are unchanged.
- No new entry arbiter or extra live configuration authority.

## Evidence boundary

This is a causal provenance correction, **not a proved strategy edge**.
The 26 October live sample is two SMC trades (one RANGE loser).
Historical RANGE samples were run under older strategy revisions and
cannot establish current conditional expectancy. Before making any
additional threshold changes, compare this version with the prior rule
using chronological futures+matching option BBO replay, all costs,
execution fill constraints and held-out days.

Release only on exact-head CI success and a broker-verified safe
automatic rollout; do not force an off-hours restart on a stale token.
