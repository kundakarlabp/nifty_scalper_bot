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

## 19 November: stop-fill and broker-lifecycle correction

- The option stop **trigger** is a bid-price threshold. Estimated adverse
  liquidation after trigger is a separate, lower price, allowing both 25-bps
  slippage stress and one additional observed spread. Both admission and the
  final one-lot runner check price the stop loss at this adverse-fill scenario;
  the protective trigger is **not** a guaranteed fill. Existing broker
  exit ownership, stop order types and time-stop remain unchanged.
- RangeVWAP now uses the existing `StrategyManager.notify_entry_accepted`
  and `notify_entry_rejected` callbacks. Only the runner's explicitly proven
  `no_affordable_execution_candidate` *before any broker attempt* releases
  a generated setup. Unknown broker results or accepted orders stay consumed.
  The existing runner cooldown and order deduplication remain authoritative.
- The in-process research funnel distinguishes **modeled** cost-eligible
  candidates, pre-broker capacity rejections and accepted orders. An accepted
  order is not a proven fill or realized return; the existing persisted trade
  journal and broker reconciliation remain the authoritative sources for
  those events. Research counts are not durable across process restarts.
- These are correctness and attribution fixes, not newly optimized entry
  thresholds or an observed positive option expectancy. Session-restart
  history recovery, historical option bid/ask calibration, contract-rotation
  replay, alternate exit policies and caching require distinct verified
  source data or benchmarks before being promoted into live decisions.

## 19 November follow-up: reuse historical evidence and time-zone provenance

- The existing `analyze_completed_trades.py` script accepts
  `--strategy range_vwap_reclaim` and isolates canonical closed Range VWAP
  trades and strategy-tagged candidate decisions without creating a second
  journal, backtester, risk model, or order path.
- A new **descriptive** `range_vwap_slippage_stress` section applies 10, 25
  and 50 basis points of *additional* adverse movement to **each** already
  observed entry/exit premium. It requires verified closed ledger, positive
  entry/exit premiums, integer quantity and broker virtual-contract-note cost
  provenance for **every** selected trade. An absent field, unverified cost
  source or zero trades produces `ready=false` and no synthetic P&L.
  This is not exchange tradebook fill parity or counterfactual bracket replay.
- The existing range futures-history recovery now uses the strategy's
  timezone-aware 09:15 anchor parser: `03:45Z` and `09:15+05:30` on the
  same date are equivalent. Stale, naive and off-minute timestamps cannot
  suppress needed recovery. The strict candle publication/continuity checks
  remain authoritative; no missing futures history is synthesized.
- Example (authorized local canonical journal export):
  `python scripts/reporting/analyze_completed_trades.py --trades-db
  /path/to/journal.sqlite --strategy range_vwap_reclaim --component
  RangeVWAPReclaim`
- As of the read-only Supabase check on 19 November, recent indexed ledger
  records contain no RangeVWAPReclaim completed trades. There is therefore
  **no basis for live threshold tuning or profitability assertions**.
- Outstanding: observed matched option order-book paths, real fill/slippage
  reconciliation and independent futures/option replay before changing entry,
  15-minute time-stop or protective exit policy. The in-memory excursion
  registry is not a durable cross-restart event store; creating a second
  persistent ownership path before confirming canonical journal coverage
  would increase duplicate-order risk.

## 19 November follow-up: same-session durability and measured-execution audit

**Live correctness changes (risk-preserving):**

- `RangeVWAPReclaimStrategy` binds to the existing `PositionManager` through
  `StrategyManager`. Emitted setup IDs, consumed futures-side-session
  excursions and still-pending broker outcomes use the same existing daily
  `risk_circuit` snapshot as other restart-safe entry controls.
- The write is verified by re-reading the canonical position state file after
  `persist_risk_circuit_state`. A failure or invalid same-day payload **blocks**
  the new RangeVWAP signal before the runner/broker sees it. No new SQLite
  store, Redis dependency or broker order path exists.
- A risk-circuit snapshot from yesterday does not seed today's idempotency.
  Unknown broker submissions remain reserved; only a proven
  `no_affordable_execution_candidate` pre-broker callback releases a
  reservation. Touching VWAP never resets a still-pending order. Concurrent
  eligible evaluations use one in-strategy reservation lock; broker dedup
  and existing active-position gates remain authoritative.
- This change does not independently reconstruct abandoned orders created
  **before** this new version was deployed; such historical uncertainty
  must continue to be handled by existing broker-state reconciliation.

**Execution calibration and exit-policy research (no LIVE retuning):**

- The existing `analyze_completed_trades.py --strategy range_vwap_reclaim`
  now reports measured *paired* entry/exit slippage only for broker-costed
  canonical completed trades with observed entry ask, observed exit bid and
  a <=2-second exit cached-quote age. It reports median and 95th-percentile
  **adverse** bps by side; experimental sample minimum is 30 such trades
  across 10 independent sessions with >=90% completed-trade coverage.
  Passing these thresholds **never** automatically authorizes new live
  settings, because broker fill reconciliation and external holdout are
  separate requirements.
- The pre-existing `exit_policy_research.py` compares 10-, 15- and
  20-minute optional time-stop quote-level overlays in addition to its
  existing target/trailing variants. It evaluates only full-size observed
  option bids captured within the *real* position's lifetime. Scenarios
  needing an observation after the actual exit or before an unresolved
  protective stop are reported UNRESOLVED, never a hypothetical profit.
- The live 15-minute time-stop, current session-VWAP targets, 25-bps
  initial stress, 1.5 after-cost R:R minimum, 773 INR modeled cash risk,
  expiry controls and broker-owned protective exits remain unchanged.
- Actual Range VWAP option fill-to-fill profitability remains **unproven**.
  The limited 17-18 November futures-only proxy does not validate option
  mean reversion after Zerodha/NSE fees, spread and adverse IV/theta.

**Research-practice implications:** VWAP is an execution benchmark, not
intrinsic equilibrium evidence. Compare reversal *vs continuation* after
pre-specified regime/volatility, option liquidity and event-period filters.
Exclude look-ahead, non-independent overlapping evaluations, wide quotes
and survivorship effects. Use a locked chronological holdout with
strategy-only net P&L, zero-trade abstention and actual broker fills before
claiming profitability or moving any exploratory threshold into production.

## 19 November: volatility-conditioned transition and option-payoff audit

The published literature supports testing *conditional* intraday behavior, not
assuming VWAP excursions must reverse. The Nasdaq VXN-breach preprint reports
non-monotone profitability across volatility regimes despite a high eventual
touch rate; prior and subsequent samples diverge. Broad futures studies report
intraday momentum during hedging-driven flow. Neither result directly validates
a 1–15-minute long NIFTY option entry: price touch, timely price touch, full-size
option bid at exit, and profitable realized after-cost fills are separate events.

This revision makes three focused changes using the existing strategy and
canonical report (no parallel signal engine or new model):

1. A *prior-completed-bar* volatility-pressure diagnostic records the last-5
   to last-14 average true range ratio and last-5-minute directional efficiency.
   It is not a new LIVE regime threshold or a probability.
2. A fresh, verified futures quote must still be inside the frozen reclaim band
   and must not have already reached exchange session VWAP. Expected futures
   travel entering the existing option delta/theta/IV/basis and post-cost
   admission model is reduced to the *smaller* of confirmation-close and latest
   futures-quote remaining travel. Invalidation risk uses the larger of the
   original and current distance to the structural stop. This prevents
   mechanically overestimating option profit from an already-progressed reversal.
3. The canonical completed-trade analysis now separates pre-entry realized
   volatility expansion (>1 versus <=1 relative to the causal 14-bar baseline),
   direction-persistence pressure, modeled **target-scenario** net reward and
   actual verified broker-costed net P&L. It explicitly reports provenance
   coverage. Target-scenario reward is **not** probability-weighted expectancy;
   selected-trade cohorts cannot label missed/rejected opportunities.

No numeric RANGE band, slope, OFI, ATR, spread, delta, cash-risk or
15-minute stop threshold is altered. The generic `TradeCandidateSelector`
already orders viable contracts by post-cost net R:R and liquidity evidence;
it is deliberately not replaced by a second, incompatible ranker. Strategy-
specific actual contract winner comparisons need *paired contemporaneous*
bid/ask, greeks, broker affordability and validated quote/OFI evidence before
additional candidate-ranking changes are justified.

Research-only next checks: out-of-sample grouped net expectancy by volatility
state and short-term directional efficiency; percentage of complete option
quote/fill evidence; forward 5/10/15-minute markouts for non-overlapping
rejected opportunities; model-to-broker spread/IV/theta and fill shortfall.
No new live threshold should be promoted on the limited futures-only proxy.

## 20 November: canonical futures-excursion provenance and executable quote research

Research-only extension, **no new LIVE admission, target, risk, trailing or broker
state owner**. Scope: the existing `RangeVWAPReclaimStrategy` can now emit
`RANGE_VWAP_EXCURSION_OBSERVED` once per selected CE/PE futures band excursion
(the first contiguous completed futures bar beyond the frozen band identifies
its causal start). The independent strategy-manager regime rejection path invokes
only its throttled observation method; it does not evaluate a broker trade.

- First immutable event records exact selected option symbol, side, futures
  contract, previous completed-bar excursion identity, regime, session VWAP,
  rolling center and ATR separation, short-horizon volatility/directional
  efficiency and actual quote data only if present and tradable.
  `RANGE_VWAP_EXCURSION_DECISION` logs distinct real rejection stages, the
  cost-eligibility stage and signal emission. Evidence is written to **existing
  journald**, copied by the existing daily archiver; no new table or collector.
- Post-market `market_research_report.build_session_research_report` produces
  exactly one canonical **first-seen** record per futures/side/excursion
  timestamp across restarts. Later option contract rotations cannot replace
  the entry option; conflicting contract identities are unlabelled.
- For 300/600/900 seconds, its report joins causal *original captured* futures
  WebSocket price paths with verified same-run, same-basket-phase, exact-token
  `ws_full` option top-of-book quotes. One-lot executable **best-level** buy
  and sell quantities are required. Sparse/gapped or stale entry/terminal
  observations are UNLABELLED, not zero, and hypothetical quotes are not fills.
- The same existing NSE/Zerodha fee model and 5-bps adverse slippage per side
  price the observed ask-to-bid *terminal* scenarios; spread is not charged
  again. Counterfactual policies are labelled
  `ignore_center_disagreement_only` and `ignore_range_width_only`.
  They qualify for a same-evaluation economics comparison ONLY if a
  verifiable modeled stop risk of **at most ₹773** and net R:R of **at least
  1.5** were present *at the first observation*. Most structurally blocked
  opportunities cannot meet this evidence bar by definition. They remain
  explicitly **risk-unverified**; do not infer an alternate profitable trade.
- None of the research outcomes establishes broker fill probability,
  executed protective-stop paths, realistic unfilled orders, signal-gate causal
  impact or out-of-sample improvement. `eligible_for_entry_filter_change`
  remains false. A 15-minute hypothetical terminal bid is not a complete
  strategy lifecycle simulation.

**Coverage caveat:** Observations require authenticated completed futures bars
and a selected option at the decision. If the session was not captured, a
strategy never evaluated, a manager context had no eligible selected option,
or a tape lacks contract-token/depth/run provenance, output is missing or
unlabelled rather than reconstructed. Volatility threshold tuning or a
"relaxed" RANGE gate remains prohibited without independent out-of-sample
execution evidence.
