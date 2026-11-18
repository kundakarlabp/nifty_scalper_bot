# NIFTY Range VWAP Reclaim — evidence, contract and promotion decision
18 November 2026 · Original research status: **SHADOW CONTEXT ONLY**

19 November 2026 · Guarded broker-live pilot code is available behind an
explicit host flag; this is **not evidence of strategy profitability or successful
production activation**. Real trading remains conditional on validated deployment,
live inputs, an active broker session and existing order protection.

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
- Compute a causal **bar-derived session VWAP estimate** using
  volume-weighted typical price of all completed bars **before the proposed
  confirmation candle**. That estimate freezes the outer band without
  future-bar leakage. Compute the 20m rolling center independently on the
  prior 20 candles. Require their divergence <=1.0 ATR and both remaining
  on the profitable side of the confirmation close.
- **Actual option target** is the fresh exchange-reported NIFTY futures
  session VWAP (`futures_vwap_source=active_future_quote_vwap`) from the
  existing cached futures quote. Require agreement with the earlier
  bar-derived estimate within 0.50 futures ATR; reject missing, divergent,
  unproven-source or stale values. Never use a synthetic spot index VWAP,
  a rolling 20m pseudo-VWAP, or a newly pulled broker quote to authorize
  a target. Because the exchange quote is read after the confirmation bar,
  it is used for *target projection only*, never to recalculate its band.
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

## 18 Nov P0–P2 follow-up: fail-closed research corrections

- Completed-band geometry no longer has an independent 0.75 ATR
  remaining-VWAP hurdle. A close must remain on the correct side of both
  centers; executable after-cost option reward is assessed separately.
- OrderFlow is evaluated first. Its same-evaluation metadata, option-contract
  identity, non-fingerprint quote version, holdoff computation and timestamp
  must reach Range VWAP intact. Missing producer evidence cannot be interpreted
  as a negative holdoff.
- The modeled option requires observed spot and futures, observed spot-referenced
  delta and explicit theta conventions. Adverse basis/IV, early-exit and
  stalled scenarios are **stress assumptions**, not calibrated premiums or
  verified fills. No broker authority was added.
- A bounded per-opportunity shadow ledger records completed futures structures
  even when subsequent quote and economic checks reject them. The session-
  anchored dispersion comparison, expiry bucket, and observed event tag are
  observational only. Missing event proof remains unknown. MFE, MAE,
  executable win rate and net expectancy remain *unmeasured* until verified
  point-in-time option books and exits are available.
- A restart cannot replace missing 09:15 history with rolling VWAP.
  The optional local DataHub recovery reads only existing 1-minute bars,
  requiring original contract, persisted-source identity, verified completion,
  and publication timestamps. Incompatible cache rows fail closed. Whether
  the currently deployed cache actually persists all those attestations must
  still be checked with runtime provenance; no successful recovery is claimed.
- This change is a research-only implementation. Source merge does not
  activate the opt-in flag or authorize orders. The modeled stop cap is not
  an upper bound on realized stop-gap losses.


## 19 November 2026: explicit real-broker pilot contract

- A dedicated pilot opt-in (`ENABLE_RANGE_VWAP_LIVE=true`) is required
  alongside actual `EXECUTION_MODE=LIVE`, a live broker permission
  (`ENABLE_LIVE=true` or `ENABLE_LIVE_TRADING=true`), and paper/shadow disabled.
  Research-only settings never silently grant real-order authority.
- The canonical taxonomy, elite builder, strategy manager and runner now agree
  on the scoped RangeVWAPReclaim trigger. All unrelated strategies preserve
  their original direction contracts. Countertrend admission is granted only to
  a **sole** verified range trigger: completed futures-band reclaim with a
  same-evaluation OFI quote, validated stressed option economics, and a checked
  premium stop/target. Fresh adverse context still vetoes the trade.
- One option lot is requested, conditional on broker affordability. The model
  admits only reward/risk >= 1.5 and estimated stop loss <= **₹773** including
  modeled costs. It never overrides broker risk, quote freshness, stale-data,
  max-loss, position/exposure or protective-exit gates. A gap or failed stop
  can lose more than ₹773; it is a model cap, not guaranteed realized loss.
- Broker enablement, observed spot/futures quote, exchange VWAP, option bid/ask,
  verified spot-referenced delta, exact expiry and contiguous completed 1m
  futures history remain mandatory. Absence of any prerequisite fails closed.
- A GitHub merge alone cannot change the host's private runtime env or prove
  production is running the new commit. Verify effective release SHA and live
  order readiness before counting any opportunity as an executable trade.
- This user-authorized pilot uses small live size rather than treating a
  backtest as a prerequisite. Performance is unknown; all fills, fees, stop
  slippage, adverse excursions and session subgroup outcomes must be audited
  retrospectively before increasing exposure.
