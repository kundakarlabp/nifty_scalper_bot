# Regime, futures-volume and first live SMC trade audit — 19 November 2026

## Verified production evidence

- Production build during live entry/exit: `e66d0126ccbc53162459b91c3786474d2d294c3a`.
- Live NIFTY 26 Nov 25350 CE, SMC, 65 units; entry 12:30:12 IST at
  INR 157.00, protective-stop exit 12:34:08 IST at INR 160.35.
- Gross P&L INR 217.75, ledger net P&L INR 146.24. Estimated costs
  about INR 71.81. One trade cannot validate strategy expectancy.
- Initial stop 151.95, target 168.40, 11 stop-trail revisions, final
  stop trigger 160.40, realized fill 160.35; favorable excursion recorded
  as 1.802R, net outcome 0.4455R. Trailing was functioning but a single
  favorable markout cannot justify changing thresholds.
- Exchange stop `2123310354569555968` was reconciled after a transient
  `MODIFY PENDING` order status; broker fill and flat position confirmed
  via trade ledger, fill event, bracket closure and broker reconcile.
- Cash-based `minimum_lot_unaffordable` while premium was reserved was
  transient: after broker flat the refresh restored available funds of
  INR 15,916.75 and execution readiness rearmed.

## Distinct regime authorities and intended policy

`MarketRegimeManager._store_snapshot()` holds two states:
- Raw: newest detector evidence (at 12:29 and 12:30, UNKNOWN, confidence
  0, reason `range_evidence_unverified`).
- Stable: last accepted hysteretic state (TREND), retained across an
  unproven UNKNOWN observation to avoid state thrashing.

`can_trade()` correctly rejects the raw UNKNOWN. In the strategy
manager's deliberately enabled adaptive mode, a raw filter rejection
does not itself veto a separately confirmed structural setup: the
retained TREND is used to route its setup, and requested sizing scale is
capped by the configured fallback scale. Previously
`signals_blocked_by_regime` counted *every raw rejection*, even when
the strategy was allowed to continue, while the runner's
`regime_blocks` counted actual vetoes. The old
`strategy_regime_scale_fallback` metadata also reported 1.0 before
computing the actual fallback. No algorithm-quality conclusion should
be inferred from either counter.

The correction records independently: raw observation, retained stable
state, whether the two agree on a verified regime, effective adaptive
policy, requested fallback scale, filter-rejection count, adaptive
continuation count, and **actual blocked entry** count. The strategy
continues using the existing stable routing policy: this patch does not
weaken a safety gate, force entries, or invent a trend.

**Research follow-up:** quantify signal outcomes during
`raw_unknown/stable_trend` separately from `raw_trend/stable_trend`
using timestamp-aligned recorded market data, then choose any expiry,
confidence, and independent-confirmation policy out of sample.
Unexpected persistent UNKNOWN should alert operators; it is not
evidence to retroactively relabel an active position.

## Futures-volume causality

The canonical futures completed-bar volume ratio is computed from
the active futures contract's verified 1-minute completed candles.
At 12:30:51 IST: ratio 1.24713, 20 baseline bars, timestamp 12:29,
reason verified. The legacy quote-counter delta ratio may be 0.0 for
the very same evaluation; it is **not** proof of zero completed-bar
volume. Completed-bar provenance requires contract identity, finality,
baseline and age checks; no synthetic volume should replace missing
data. The strategy's VWAP-specific confirmation event is reached only
after its earlier distance checks. A separate upstream event now
reports evidence before a VWAP strategy can return early, and
minute-sampled research rejections carry both ratios and the regime
provenance as different fields.

## Connectivity, execution and unresolved operating requirements

- Heartbeat reported engine healthy, exchange data fresh and WebSocket
  polling fallback inactive. Occasional slow tick/dispatch warnings
  were observed (~211 ms peak in earlier post-deployment logs), but
  there was no demonstrated sustained market-data interruption.
  Tune only against latency histograms, p95/p99 and dropped ticks;
  preserve nonblocking broker reconciler and existing stale-feed gate.
- The AppDeploy diagnostics relay still reported HTTP 404
  `Backend not found`; this does **not** establish engine failure.
  The Supabase reporter independently verified engine and trades.
- NSE options stop-loss market orders are not allowed; the exchange
  stop was a stop-limit order, which is *not guaranteed to execute* if
  the market gaps through its limit. Preserve exit retry, order-status
  reconciliation, rate limits and emergency flat controls.
- The operator's stated absolute daily-loss ceiling of INR 773.00 is
  not verified in running config. Five percent of INR 15,699.00 is
  INR 784.95 and five percent of INR 15,916.75 is INR 795.84.
  Existing risk code can enforce the smaller of the percentage cap
  and the positive absolute cap with `RISK__MAX_DAY_LOSS=773.0`.
  Updating source/example config **cannot change production env**.
  Apply a broker-safe operator cutover after independently checking
  orders, stops, day P&L and frozen cap; verify the live risk snapshot.
- One winning trade with a favorable excursion cannot determine the
  optimal SMC stop, trailing tiers, VWAP extension, or spread cutoffs.
  Compare protected exits, lost opportunity, adverse selection, and
  fee-inclusive realized R across multiple regimes using forward
  walk-forward/out-of-sample periods before promotion.

## References (context, not claims of demonstrated profitability)

- QuantConnect LEAN Algorithm Framework (official): separate Alpha,
  Portfolio Construction, Risk Management, Execution and fill state.
  https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview
- Zerodha option stop-order constraints:
  https://support.zerodha.com/category/trading-and-markets/alerts-and-nudges/kite-error-messages/articles/im-trying-to-place-a-stop-loss-order-but-im-getting-the-error-limit-price-is-worse-than-the-trigger
- Bailey et al., *The Probability of Backtest Overfitting*:
  https://escholarship.org/uc/item/4w1110bb

All changes in this patch are observability/policy-accounting and tests;
trade execution, order APIs and protective exits are unchanged.
