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

## Empirical available dataset

- Owner's Google Drive holds 25 CMS-encrypted original Breeze windows from
  6 November 2024 to 5 November 2026; prior archive audit reported 1,454,867
  one-minute index/futures/selected-option OHLCV observations. Original private
  key not available in this connected environment, and archives have not been
  newly decrypted/verified and replayed here. No raw option bid/ask/depth.
- Distinct validated EOD catalog contains ~494 F&O sessions but no synchronized
  1m execution tape or option quotes. It cannot estimate one-minute reversal
  entry/exit or point-in-time instrument selection.
- 18 November 2026 live journals contain sparse NIFTY futures 5-minute price/
  exchange-VWAP observations (31-32 snapshots). In the 23 overlapping samples
  with futures at least 20 points below their exchange VWAP, the following
  15-minute price move was positive in 14 and negative in 9. Median +4.5 future
  points, min -50.3 and max +75.6. These overlap and omit full opening hours;
  they are NOT 23 trades, NOT validated reversal triggers and NOT options P&L.
- At 12:17 and 13:55 two distinct SMC CE sweeps suffered adverse premium
  movement before possible rebound; 25bps stress on sampled option bid/ask
  marks show why a countertrend strategy must first prove completed-bar
  invalidation, cash risk, source authenticity and protective execution.

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
* **Not knife-catching:** prior closed futures candle touches the excursion,
  then current fully closed candle reverses without setting a worse low/high,
  while leaving at least 0.2 futures ATR to the rolling center.
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

## Reproducible owner-only research

1. On trusted owner machine, locate matching existing CMS private key without
   sharing it; use `scripts.finish_breeze_private_replay` as described in
   `docs/research/breeze_sqlite_offline_research_handoff_20261113.md`.
   Do not download/re-encrypt the original 25 windows again.
2. Export a **real contiguous one-minute dated NIFTY futures** series from the
   verified private minute SQLite; preserve `start_ist`,
   `available_at_ist`, OHLCV. Never assume timestamp semantics.
3. For one exported file run:
   `python -m scripts.reporting.range_vwap_reclaim_backtest
   --futures-csv /private/futures_1m.csv --output /private/range_results.json`.
   The report records setup coverage but explicitly returns no trade P&L.
4. Upgrade to genuine options replay only with matching dated contract
   master, exact lot, synchronized point-in-time option bid/ask (with spread/
   depth) and timestamped protective-order latency. Use canonical
   `backtesting/runtime_research.py` for recorded production sessions,
   then walk-forward development/validation/untouched later holdout,
   overlapping signal de-duplication, 10/25/50bps fill stresses and fees.
5. Promotion requires adequate independently resolved trade count, positive
   after-cost expectancy with confidence intervals, stable PF/drawdown,
   99th-percentile stop slippage and broker reconciliation evidence, and
   prospective SHADOW logs. Never promote from a single profitable day or
   optimistic OHLC midpoint fills.

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
