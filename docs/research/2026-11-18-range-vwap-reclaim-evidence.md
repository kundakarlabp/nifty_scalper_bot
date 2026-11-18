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

## Proposed range-only elite strategy

`strategies/elite_strategies/range_vwap_reclaim.py` is a distinct **context**
member of the existing elite catalog. It does not change VWAPPro / SMC / ORB
admission, manager arbitration, live exposure, stops, routing or risk switches.

* Explicit opt-in: `ENABLE_RANGE_VWAP_RESEARCH=true`. Default off. The
  strategy is context-only in canonical taxonomy. In `EXECUTION_MODE=LIVE`,
  it NEVER returns a vote, regardless of the opt-in setting.
* Source: 40 consecutive complete 1m candles from canonical NIFTY dated
  futures, same IST session, valid OHLCV and positive volume; last confirmed
  candle no older than 90 seconds, zero bar/candle interpolation and no
  imputed bid/ask. Canonical futures symbol is taken from the existing
  instrument resolver.
* Proposed uncalibrated research parameters: rolling last-20 volume-weighted
  typical price; ATR14; excursion min max(0.8 ATR, 1.1 close sigma);
  VWAP slope <=1.2 bps/min over 5 min; realized net-direction/total movement
  efficiency <=0.35; 30-bar width 1.5–6 ATR; confirmation volume relative
  to prior 20 candles >=0.85; no final impulsive bar >2 ATR.
* Decision window: 10:00–14:30 IST, excluding the opening and closing volatility zones.
* **Not knife-catching:** prior closed futures candle touches the excursion,
  then current fully closed candle reverses without setting a worse low/high,
  while leaving at least 0.75 futures ATR to the rolling center.
* Uses underlying canonical CE/PE agreement without a countertrend override,
  freshness <=2s and existing selected-option eligibility, executable bid/ask,
  quote age <=1.5s, observed dated contract lot size and option ATR.
* Research payoff floor: model premium stop max(0.75 option ATR, 3.5% ask),
  aspirational gross target 2.5 stop distances, at least 1.5 **modeled**
  after-cost target-to-stop R:R; cash-loss budget <=₹773. Shared Zerodha/NSE
  brokerage, STT, GST, stamp, exchange and SEBI model; 25bps extra adverse
  entry and exit slippage. This is geometry, **not expected realized edge**.

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
This is evidence-driven opt-in safe implementation, not a proven profitable
strategy. The ₹773 cap in the research model does not change the production
per-trade 5% risk setting; any discrepancy must be corrected independently
after broker-state and deployment safety validation.
