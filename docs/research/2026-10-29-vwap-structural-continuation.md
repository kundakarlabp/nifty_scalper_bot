# 29 October 2026 — causal VWAP continuation-quality research

## Purpose and evidence

Two realized losing entries motivated a causal investigation, not a fitted threshold. The
29 October VWAPPro continuation was admitted without independent trigger consensus
after an earlier, unfilled attempt into the same setup. The 26 October SMC loss had
a stale structural sweep; its original-sweep age is already enforced separately.
These cases do not establish a winning threshold or prove OrderFlow has no value.

External references: [TradingView confirmed bars](https://www.tradingview.com/pine-script-docs/concepts/repainting/);
[NinjaTrader OnBarUpdate](https://ninjatrader.com/support/helpguides/nt8/onbarupdate.htm);
[QuantConnect algorithm separation](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview);
[Kite FULL depth](https://kite.trade/docs/connect/v3/websocket/).

## Implementation

`VWAP_STRICT_CONTINUATION_ENABLED=true` is default-on and applies specifically
to `vwap_continuation` (other setup subtypes retain their original rules).
The strategy uses completed observation-bar timestamps deterministically.
The existing `strategy_live_safety` boundary separately checks the original
thesis anchor against the actual LIVE decision clock (180-second cap), rejecting
missing, future or expired proof. A fresh signal-bar timestamp cannot revive
an old live setup. This keeps historical replay and LIVE safety separate.
When enabled, the existing completed option bar proof is required. Continuation
must have a same-session thesis no older than 180 seconds **at the completed
signal bar**, contiguous finalized previous option candle with trades, confirmed
TREND regime, fresh underlying direction at the existing early-trend threshold,
a close above the prior candle high, close in the upper 30% of the candle,
contemporaneous option volume ratio at least 1.0, and option-VWAP distance no
more than the existing base quality ATR cap. In addition it requires either:

1. Retest/rebreak: prior candle touched its *own timestamp-aligned* session
   VWAP, closed above it, current candle held current VWAP, and closed bullish; or
2. Momentum breakout: current bullish body exceeds the existing 0.35 ATR
   momentum threshold and its close exceeds previous candle close.

The current OrderFlow context vote and adverse-flow veto **remain unchanged**,
as do quote, risk, sizing, broker, stops, and exits. This is intentionally not
another strategy or duplicated signal-scoring system. The emitted metadata
reports opt-in state and qualifying proof (when approved).

## Research promotion, not live approval

- Run the canonical focused/full regression gates and inspect exact-head CI.
- Replay original and research policies on chronologically aligned completed
  option bars, futures/spot context, option depth, bid/ask, events and broker
  state, especially 26/29 October and rejected opportunities.
- Compare after-cost net expectancy, maximum drawdown, trade frequency,
  false-breakout rate, missed winners, fill latency and sensitivity around
  the 180-second age and 1.0 volume-ratio hypotheses.
- Perform a final chronological holdout and shadow/paper evaluation before a
  broker-flat live-small release. Do not infer that delayed confirmation is
  always safer: it may worsen entry prices.
- Keep the unmerged same-setup entry-chase PR #1917 separate until its
  formatting, broker reconciliation and CI gates pass.
- The quality gate is default-on in source; rollout must retain the existing
broker-flat release and monitoring safeguards. Candidate is
`ENGINEERING_CORRECT` only once full CI passes; no demonstrated alpha.

## 29 October default-on operational decision

The operator explicitly selected default-on structural qualification. The same
variable controls both VWAPPro's strategy-owned finalized-breakout rule and
the canonical live-safety original-thesis timeout; they cannot drift by default.
An explicit `false` value is the rollback. This does **not** enable the broker,
change execution mode, or weaken risk, stops, or position reconciliation.

All directional live triggers are mapped once in the canonical builder:
VWAPPro uses completed premium breakout/retest or momentum continuation;
SMC requires completed futures sweep/displacement and independent existing
volume or timestamped BOS/CHOCH evidence (or paired retest and premium reclaim);
ORBPro already requires completed futures range, volume/penetration, and
momentum or a held retest. OrderFlow remains context-only; other analysis
providers cannot authorize new entries. Expiry gamma is isolated by
`STRATEGY_MODE=expiry_gamma` and `ALLOW_EXPIRY_GAMMA_STRATEGIES`.

The separate PR #1917 remains unmerged because its broker-submit façade
wrapper violates the single-runtime-owner architecture contract. This change
does not adopt or rely on that implementation.
