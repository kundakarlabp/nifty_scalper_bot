# NIFTY Range VWAP Reclaim — evidence, contract and promotion decision
18 November 2026 · Research status: **SHADOW CONTEXT ONLY · NO LIVE ENTRY**

## Evidence and inference

VWAP measures executed volume-weighted price, not conditional future mean
reversion. Short-horizon returns show regime-dependent reversals, continuation,
volatility clustering and bid-ask bounce; a VWAP distance is not a standalone
profitable trade signal. Studies of VWAP execution benchmarks do not validate
the P&L of buying Indian weekly options after a futures VWAP excursion.

Relevant background:

- Hendershott, Jones & Menkveld, *Does Algorithmic Trading Improve Liquidity?*
  Journal of Finance (2011), DOI: 10.1111/j.1540-6261.2010.01573.x
  (liquidity and execution, **not** range-alpha proof).
- Intraday return patterns and short-horizon reversals can depend on liquidity,
  risk, spread and event-time context. Cross-market empirical results must not
  be transferred as calibrated NIFTY option thresholds.
- SEBI, *Study on Profitability of Individual Traders in the Equity Derivatives
  Segment*, August 2026: https://www.sebi.gov.in/ (regulatory research). Large
  losses among retail F&O participants warrant a higher after-cost bar.

## Accessible two-session broad replay — deliberately NOT an options backtest

**Breeze research excluded by operator request.** We used only the existing
read-only production log archive (`nifty_daily_market_logs`) from 17–18
November 2026, no new data acquisition or paid feed. The archived
`FUTURES_VWAP_PROVENANCE` records contained 120 five-minute futures
LTP/session-exchange-VWAP observations: 75 on 17 Nov, 45 on 18 Nov.

For each observation with `abs(futures_ltp - futures_vwap) >= 20` points,
measure the signed *next-observed 15-minute* futures markout, long futures
when under VWAP and short futures when above VWAP. Match a same-day observation
between 14m55s and 16m after entry. No observed candle lows/highs or valid
option fills exist in this snapshot-only stream.

| Date | Overlapping eligible snapshots | Toward-VWAP marks | Mean signed futures move | Worst signed move | Nonoverlapping subsample |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 Nov | 55 | 26 | +3.41 points | -54.3 points | 18, of which 7 positive |
| 18 Nov | 26 | 17 | +3.75 points | -50.3 points | 8, of which 5 positive |

These are 5-minute snapshots and **a naïve VWAP-distance negative control**,
not qualifying RangeVWAPReclaim completed-bar setups, 81 independent entries,
or P&L from trading options. Positive gross means near 3–4 futures points are
small and unstable compared with adverse moves; this sample provides **no
credible after-cost NIFTY option expectancy**. Five-minute data cannot
demonstrate a consecutive 40 x 1-minute futures history, completed-bar low/
high rejection, entry-after-confirmation or quote-side stop execution. Thus
strategy backtest eligible setups = NOT OBSERVABLE, realized option net P&L =
NOT ESTIMABLE, live promotion = FALSE. The original two SMC rejections
around 12:17 and 13:55 on 18 Nov likewise showed adverse quote excursions,
not proofs of range alpha.

**No parameter optimization or threshold curve fitting** was undertaken on
these two sessions. Treat the thresholds below as conservative hypotheses
for future prospective recording; choose them on separate development
data and evaluate on unseen later sessions before broker admission.

## Revision: session VWAP as primary, no local-range-rotation trades

The range strategy exclusively targets **09:15-anchored NIFTY futures session
VWAP**. It must not use the rolling 20m volume-weighted center as the actual
VWAP target. The rolling center is retained only as independent corroboration.

### Causal reference and entry sequence

- Obtain up to 400 most recent IndicatorEngine bars for the canonically
  resolved NIFTY futures; verify complete, non-provisional, correctly aligned
  1-minute OHLCV **continuously from 09:15** (minimum 40 bars). Never silently
  substitute a 40-minute rolling center when a session starts late. Incomplete
  or gapped history => no research candidate.
- Compute session VWAP using volume-weighted typical price of **all completed
  bars before the proposed confirmation candle**. Compute the 20m rolling
  center independently on the prior 20 candles. Require their divergence
  <=1.0 ATR and both remaining on the profitable side of the confirmation
  close. Session VWAP is the price target, not the rolling center.
- Outer band is fixed **before** the confirmation bar: max(0.8 ATR14,
  1.1 population standard deviations of the prior 20 closes). Require the
  prior candle's close and extreme beyond that band, then an adjacent fully
  completed candle which closes back **inside** by at least max(0.10 ATR,
  5% of band width), with >=30% real body, no worse low/high, adequate
  prior relative futures volume (>=0.85) and no impulsive >2 ATR candle.
- No mechanical knife catching. Exclude a sloping session VWAP (>1.2bps
  per minute), directional efficiency >0.35 and 30-bar range width outside
  1.5–6 ATR. Require >=0.75 futures ATR remaining to BOTH centers.
- Keep entry window 10:00–14:30 IST. Canonical RANGE ontology normalizes
  CHOPPY/RANGING aliases. Require two **chronologically adjacent** observed
  completed bars in stable RANGE; duplicate ticks within a bar do not count.
  Reject regime transitions, stale option/underlying quotes and spot/futures
  directional conflict. The legacy trend bias need not agree with the
  new two-completed-candle reversal **for research context only**.

### Option and execution evidence (research geometry, not proven alpha)

- Require a verified, near-ATM/selected valid CE or PE contract, observed
  exchange lot size, bid/ask spread <=0.40%, context age <=2s, option quote
  age <=1.5s, and verified signed delta 0.25–0.85 absolute.
- Require fresh, sustained positive **option-premium** OFI (3s window with
  >=3 updates, >=1.5s elapsed coverage, normalized >=0.10, nonadverse latest
  1s) and no reported OFI reversal holdoff; reject explicitly adverse observed
  option-book depth imbalance (<= -0.15). Both CE and PE are bought at
  the option ask: positive OFI is supportive for either long-premium side.
- Ignore the old trend-bias direction **only inside independent structural
  research**. The production strategy manager STILL enforces continuation-only
  live trade admission; no approval bypass is introduced.
- Expiry day and unavailable expiry classification are excluded (minimum
  1 remaining day). Compare other expiry-dependent effects later.
- Model maximum *reachable* option gain at
  `abs(observed delta) * session-VWAP futures distance * 0.80`, subtract
  observed adverse daily theta for an assumed 15/375-minute hold, or deduct
  1% of option premium when theta is unavailable. Then cap by 2.5 × stop
  points; do not invent positive gamma, IV expansion or an arbitrary 2.5R
  target as an automatically attainable outcome.
- Premium stop >= max(0.75 option ATR, 3.5% ask,
  1.10 × abs(delta) × futures structural-invalidation distance).
  Buy at ask +25bps and sell at projected bid (extra observed spread and
  -25bps). Deduct the repo's broker, STT, GST, exchange and stamp cost model.
  Require after-cost net target/stop reward-risk >=1.5, stop-risk <=₹773.
  Stop fill may exceed modeled loss; no protective order is created here.
- One candidate per session excursion per side. Do not rearm until a subsequent
  completed futures candle returns within 0.25 ATR of session VWAP. Research
  setup identity is anchored to the specific preceding excursion candle.

**No local-range support/resistance rotation variant** is introduced.
Range VWAP Reclaim remains context-only and default disabled. It returns
no eligible order vote in LIVE mode and does not amend the manager, broker
safety, or the operator risk cap. Proposed thresholds are engineering
hypotheses, not statistically selected profitable settings.

## Rejection and proof boundaries

Eligibility fails on stale/ambiguous/incomplete futures bars, missing or
conflicting quote, unverified lot size, unsupported directional state, wide
spread, low volume, impulsive range break, excessive directional efficiency,
unfavorable after-cost R:R or excessive one-lot cash risk. Existing live
continuation-only manager gate is intentionally not bypassed.

**No activation:** Option buying in a range requires broker-reachable,
futures-derived, historically causal option pricing. A futures-only backtest
with no historical option bid/ask/stop-fill telemetry MUST report net P&L and
profit factor as null and `live_promotion_eligible=false`.

## Reproducible feasible research — without Breeze

1. Review the existing read-only `nifty_daily_market_logs` futures provenance
   records. Deduplicate each date/timestamp and match same-session 15-minute
   forward futures observations; report sign-adjusted excursions, overlap,
   worst adverse movement and absence of options execution evidence.
   **This limited replay was completed above.**
2. `scripts/reporting/range_vwap_reclaim_backtest.py` can evaluate genuine
   **other-source** completed 1-minute NIFTY futures CSV files with
   `start_ist,available_at_ist,open,high,low,close,volume` fields.
   It reports verified futures setups and **no option trade P&L**.
   No such 1-minute external dataset was supplied or independently verified
   for the present two-day check. Do not interpolate 5-minute snapshots into
   1-minute OHLC.
3. Keep the new elite strategy disabled for LIVE triggering. For future
   recorded-feed evaluation, require causal matched futures/spot and option
   bid/ask, verified contract lot, broker-protective stop path, fee/slippage
   stress at 10/25/50bps and later untouched test days. Do not depend on
   inaccessible encrypted data as a precondition for research.
4. An independently verified cost-adjusted positive expectancy on sufficient
   future trades, with zero unresolved broker-stop states and strictly
   observed executable prices, is required before changing the manager's
   continuation-only admission authority.

## Legacy cleanup

The standalone `src/nifty_scalper_bot/strategies/vwap_mean_reversion.py`
utility and its superseded unit tests are retired. A **separate** legacy
class in `src/nifty_scalper_bot/strategies/signal_generator.py` is kept
untouched for compatibility; do not indiscriminately remove it.

## Disposition

**Add to elite catalog, research only. NO live order enablement.**
This is a prospective research implementation, not a proven profitable
strategy. The two-session futures-only proxy cannot validate these new rules. The ₹773 cap in the research model does not change the production
per-trade 5% risk setting; any discrepancy must be corrected independently
after broker-state and deployment safety validation.
