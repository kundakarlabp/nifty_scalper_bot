# VWAP Strategy Audit (In-Depth)

## VWAP historical-recovery correction — 2026-09-11 (Asia/Kolkata)

Source baseline: `f1ef2af78e4d85c487595d2111b5d6f8294352a5`.
The moving-VWAP recovery concern is reproduced and corrected. Previously, old
option candles were compared with the latest VWAP, so later session movement
could retroactively create a reclaim thesis. Recovery now reconstructs the
session's cumulative option VWAP in chronological order and compares every
completed candle only with the VWAP observable at that candle. The newest valid
touch inside the configured recovery lookback owns the setup anchor.

Acceptance evidence: the forward-looking fixture failed before the correction;
all 18 restart/identity tests, the strategy/context suites, backtest/replay
suites, compilation, and a clean full-suite rerun passed. One unrelated partial
fill simulation failed during the first full run, passed in isolation, and
passed in the clean rerun. No threshold, sizing, stop, execution mode, or live
order behavior was changed. Engineering owner: repository maintainer/assistant;
next review is remote CI and unchanged-head merge. No external deadline.

## VWAP timestamp identity correction — 2026-09-10 (Asia/Kolkata)

Source baseline: `64ef657a293ef1d6256797a486dd89bbd486eccb`.
The restart-identity concern below is now reproduced and corrected locally.
Four RED cases showed equivalent ISO/epoch bar timestamps generating different
setup IDs after recovery. One strategy-local normalizer now owns live reset,
reclaim and history-recovery anchor representation. Fourteen cases cover both
arming paths, seven timestamp representations, and downstream deterministic IDs.
The existing UTC datetime string representation is preserved, matching
`IndicatorEngine` bar timestamps and the `signal_generator` timestamp source.
Naive timestamps follow the indicator engine's UTC convention. Opaque legacy
anchors retain their text fallback; no wall-clock timestamp is invented.

Rollout limitation: previously persisted noncanonical IDs are not rewritten.
Prefer a between-session rollout if a running producer emits ISO/epoch anchors;
no production producer format or deployed SHA was verified during this change.
Historical-versus-current VWAP selection is resolved in the dated correction above.
No entry thresholds, session/contract scopes, risk policy or order path change.
Engineering owner: repository maintainer/assistant. Required validation and merge
evidence are recorded in the associated PR; next review is deployed identity
pairing, followed by a moving-VWAP recovery regression. No external deadline.

## Current strategy/execution review — 2026-09-10 (Asia/Kolkata)

Source baseline: `cf330a1f3605cd8d3d66d494ab6f389aa7aaad58`.
This dated review supersedes the historical narrative below. In particular,
current VWAPPro uses option-premium VWAP/ATR; it does not require futures/index
VWAP or implement the historical z-score/multi-bar acceptance description.
No current production logs, deployed SHA or net-cost profitability were verified
in this review. Code correctness is not evidence of trading alpha.

| Priority | Finding and source | Decision/status | Next review |
| --- | --- | --- | --- |
| P0 | `order_manager_core.place_order` rejects single-position conflicts before acquiring a reservation. The public `order_manager` wrapper then calls `entry_geometry.release_prebroker_entry_reservation`, which could delete the preceding entry's reservation. | Reproduced for distinct and repeated setup identities. Four-line native correction preserves the reservation on `single_position_gate:` rejection. | Required CI and merge; subsequently verify deployed SHA and reservation lifecycle in logs. |
| P1 | `vwap_pro._evaluate_signal` now uses explicit underlying direction for alignment; PR #1215. | Already merged at the source baseline; retain generic fallback and existing confidence/freshness rules. | Replay accepted/rejected candidates with conflicting direction fields. |
| P1 | `vwap_pro._recover_thesis_anchor_from_history` compared historical OHLC with the latest VWAP. | Reproduced and corrected on 2026-09-11 using contemporaneous cumulative session VWAP; timestamp normalization remains preserved. | Required CI and merge, then verify recovered setup IDs in deployed logs. |
| P1 | SMC uses the underlying sweep timestamp/source for setup identity; ORB checks exact opening-minute coverage. | Preserve these already merged corrections. | Replay restart, duplicate bars, synthetic bars and ATM rotation with recorded provenance. |
| P2 | VWAP early-trend pullback requires `not premium_above_vwap`, but below-VWAP closes return earlier. | Confirmed unreachable branch. Do not enable additional entries without separate replay evidence. | Decide removal versus explicitly specified opt-in behavior. |
| P2 | Cleanup still reads shared `_last_order_decision`; other early-return and concurrent ownership paths need review. | Broader lifecycle audit remains open; this correction covers the reproduced single-position-gate path only. | Test stale decisions, incomplete broker-attempt evidence and ownership replacement under lock. |

Engineering owner: repository maintainer with assistant implementation support.
No external deadline was supplied. Review order is event-based: finish required
CI/merge before starting another production correction. No live orders are used
for validation, and no risk limit, entry threshold or stop geometry is relaxed.

Local verification: two new cases failed before the fix; all 16 focused tests
passed afterward. Full pytest and separate execution/risk/E2E/architecture suites
passed with `PYTHONPATH=src TZ=Asia/Kolkata BRACKET_AUTO_RESTORE=false`; the full
suite had one existing skip and emitted a background closed-log-stream warning.
Production compilation and helper mypy passed. Baseline comparison found no new
Ruff/Black debt. Required remote CI and merge evidence are recorded in the PR.

Source register and evidence boundary:

- Complete VWAP evaluation/recovery functions, public execution facade,
  reservation helper, runtime `place_order`, and core decision/gate ordering
  provide the direct behavioral evidence. Focused regressions exercise the
  public cleanup wrapper with the core gate's exact decision contract.
- [Kite order documentation](https://kite.trade/docs/connect/v3/orders/), reviewed
  2026-09-10: placement acknowledgement does not establish execution; order
  history/status and asynchronous updates own broker truth. Preserving pending
  entry protection is consistent with that distinction.
- [OCC/OIC option price behavior](https://www.optionseducation.org/referencelibrary/faq/option-price-behavior),
  reviewed 2026-09-10: premiums depend on underlying price, volatility, time and
  other inputs. This supports separating directional context from premium
  execution geometry, not a profitability claim for any particular VWAP rule.
- Parameter changes require timestamped underlying/futures bars, selected option
  bid/ask and depth, setup IDs, candidate/result pairs, fills and realized costs.
  Evaluate incremental net expectancy, drawdown and turnover on chronological
  holdout periods before considering live parameter changes.

## Historical narrative — retained for provenance, not current specifications

## Scope and intent
This audit focuses on the VWAP-driven strategy stack and its data/telemetry flow. The goal is **maximum diagnostic clarity** with **minimal code change** and **zero regressions**, while preserving **strategy logic, thresholds, and execution order**.

## Strategy inventory (code scope)
- **VWAP Pro (elite strategy)**: `src/nifty_scalper_bot/strategies/elite_strategies/vwap_pro.py`
- **VWAP mean reversion (signals)**: `src/nifty_scalper_bot/strategies/vwap_mean_reversion.py`
- **Primary strategy orchestration & filters**: `src/nifty_scalper_bot/strategies/runner.py`, `src/nifty_scalper_bot/strategies/signal_generator.py`
- **VWAP indicator computation**: `src/nifty_scalper_bot/strategies/indicators.py`
- **Market regime guardrails**: `src/nifty_scalper_bot/analytics/regime_gate.py`
- **Market data & VWAP sourcing**: `src/nifty_scalper_bot/streaming/polling_streamer.py`, `src/nifty_scalper_bot/streaming/kite_ticker_streamer.py`, `src/nifty_scalper_bot/core/app.py`

## Literature and industry alignment (high-level)
VWAP strategies are widely referenced as **intraday fair value anchors** in both academic and practitioner literature. Common themes include:
- **Institutional execution benchmarks**: VWAP is frequently used as a benchmark for best execution, particularly for intraday flows (e.g., VWAP as a performance yardstick in trade execution research).
- **Mean reversion & pullback mechanics**: Many trading platforms describe VWAP-based pullbacks and mean reversion around VWAP as a core intraday strategy archetype (e.g., VWAP bands, standard deviation envelopes, and reversion triggers).
- **Liquidity and regime sensitivity**: VWAP signals are generally more reliable in liquid, normal-volatility regimes; many systems add volume gating and regime filters to avoid false signals in low-liquidity or trend-dominant sessions.
- **Industry implementations**: Practitioners on major platforms (QuantConnect, TradingView, NinjaTrader, and broker execution toolkits) emphasize consistent VWAP calculation, robust volume inclusion, and avoiding VWAP use when volume or price integrity is poor.

This repository broadly mirrors those principles with:
- Volume-gated entry checks
- VWAP confluence with futures/index bias
- Over-extension filters
- Regime gating integration

## VWAP data sourcing and reliability (diagnostic clarity)
Key observations on data flow and integrity:
1. **VWAP retrieval is multi-layered**:
   - `polling_streamer.py` explicitly attempts to source `average_price` (VWAP) and logs whether VWAP/volume exist.
   - `kite_ticker_streamer.py` normalizes the `average_price` field into `vwap`.
   - `core/app.py` ensures a `vwap` key exists even when missing from quotes.
2. **Signal generation uses index/futures VWAP as bias**:
   - `vwap_pro.py` hard-blocks if index/futures VWAP is missing or zero, protecting against mis-sourced spot VWAP.
3. **Fallback behavior is present elsewhere**:
   - The broader runner includes fallback strategy behavior when VWAP is zero, while VWAP Pro explicitly blocks without futures/index VWAP.

**Diagnostic emphasis**: The system already contains defensive guardrails around VWAP presence and validity. These are consistent with industry best practices that emphasize VWAP reliability only when volume and quote integrity are strong.

## Strategy logic alignment review
### VWAP Pro (elite strategy)
- **Direction bias**: Requires futures/index VWAP for bias confirmation and blocks signals if unavailable.
- **Acceptance gating**: Enforces multi-bar acceptance, resets on cooldown/volume/strike lock violations.
- **Over-extension filter**: Uses VWAP standard deviation (if available) to reject extreme z-score entries.
- **ATR-based risk framing**: Applies consistent SL/TP based on option premium behavior, not index direction.

### VWAP Mean Reversion (signal utility)
- Implements direct VWAP deviations with explicit checks for missing bars/zero volume.
- Emits metadata that is suitable for downstream diagnostics.

### Runner & signal generator
- Enforces VWAP-based filters at the global signal stage.
- Uses VWAP cross-over gating and momentum fallback only when VWAP is absent.

## Observed strengths
- **Robust VWAP presence validation** before signal gating.
- **Futures/index VWAP preference** to avoid spot VWAP anomalies.
- **Explicit anti-noise mechanisms**: multi-bar acceptance, cooldowns, volume gating, and z-score over-extension checks.
- **Risk framing** uses option premium directionality (critical for options), which aligns with production-grade strategy safety practices.

## Potential risks and diagnostic gaps (no logic changes implied)
1. **VWAP source fragmentation**: VWAP sourcing flows through multiple components; diagnosing mismatches may require tracing `average_price` → `vwap` across modules. A single consolidated data provenance note or diagnostic map would speed investigations.
2. **Market data integrity dependencies**: Strategy behavior depends heavily on volume, VWAP standard deviation, and index/futures VWAP fields. Any upstream quote or volume degradation can block signals or skew confidence.
3. **Stateful acceptance counters**: Acceptance counters are keyed by symbol and direction; if symbols recycle (expiry rollovers), the counters reset via key differences, which is expected but can cause transient behavior near expiry transitions.

## Recommendations (non-invasive, no logic changes)
1. **Document VWAP provenance and requirements**: Add a short operator guide describing where VWAP is sourced, what fields must be present, and which modules enforce hard-blocks. This reduces mean-time-to-diagnosis when VWAP is missing or zero.
2. **Add a VWAP diagnostic checklist**: Provide a short checklist (quote fields, volume, index vs futures VWAP availability) to confirm signal eligibility.
3. **Expand runbook visibility**: Include the expected signal gating flow and common rejection causes in the runbook to speed on-call response.

## Conclusion
The VWAP strategy stack is generally aligned with global best practices: it prioritizes reliable VWAP sourcing, volume gating, and regime awareness, and it applies risk controls that respect option premium mechanics. The most impactful improvements for production reliability are **diagnostic documentation** and **data provenance clarity**, rather than changes to strategy logic.


## 2026-10-18 targeted follow-up: causal replay and optional completed-bar entry proof

Repository baseline: `a7fd5db7b2497011a0c79c329fe9d51c840fbdca`.

### Confirmed engineering corrections

- VWAP thesis restart recovery now sorts completed, same-session option bars by
  normalized observation time before calculating cumulative typical-price VWAP.
  Reversed broker history must not manufacture an earlier VWAP crossing.
- Non-finite OHLC or volume and negative volume are skipped before they can
  poison the session cumulative volume/turnover. Missing or zero-volume input
  retains its previous treatment. Recovery remains bounded by the current
  observation timestamp and only the same contract/session can rearm.
- A single finalized-option-bar proof block is reused by the existing
  `VWAP_TREND_MOMENTUM_ENABLED` profile and a new independent, **default-off**
  `VWAP_REQUIRE_FINALIZED_SIGNAL_BAR` research switch. The proof checks
  timestamp, open, close, candle completion, plausible high/low and finite
  premium volume. In default LIVE mode, the original entry criteria remain
  unchanged. The candidate records both required/confirmed status.

### Behaviour boundaries and research experiments

No threshold, score weight, contract selection, stop, target, risk gate, order
placement, execution mode, broker state or production environment is changed.
`VWAP_PROXIMITY_PCT` remains telemetry-only, `VWAP_EMA_PERIOD` remains
a history floor, and the configured 2R target remains a **proposal** rather
than a calibrated win probability. The opt-in check should not be promoted
until matched same-contract minute bars are proven available at the decision
timestamp and baseline-vs-opt-in replay is positive after costs.

Run matched strategy backtests, using the existing experiment tracking and
fee/depth model, for (1) completed-bar versus existing reclaim,
(2) immediate reclaim versus retest, (3) stricter time-aligned option/futures
participation, (4) option-premium distance caps, (5) bounded time exits and
current targets. Compare all accepted and rejected setups using aligned
opportunities, chronological session/expiry splits and a reserved holdout.
None of these **unvalidated parameter changes** is enabled by this patch.

### Regression acceptance

New tests cover deliberately unsorted session history, non-finite price and
volume poisoning, opt-in absence of completed-bar proof, positive CE/PE
completed-bar acceptance, and malformed OHLC. Run focused VWAP suites, full
repository validation and final-head PR CI before main merge. CI results
are recorded on the linked PR; tests alone do not prove trading profitability.


## 2026-10-18 next batch: recovery history integrity

The next RED-first regression identifies another restart-only safety problem.
Repeated completed same-minute option bars were counted repeatedly in the
cumulative option VWAP, sometimes creating a false touch and a different
setup ID. Other corrupt histories permitted incompatible OHLC geometry or
a zero-volume candle to count as a VWAP touch.

The existing chronological recovery owner in `vwap_pro.py` now:

- Deduplicates identical completed rows by normalized UTC minute timestamp,
  including equivalent timestamp formats, before applying volume weights.
- Rejects the *entire historical recovery* when two otherwise eligible
  completed rows disagree on the same timestamp, rather than selecting an
  arbitrary corrected version; normal intraday evaluation remains unchanged.
- Ignores impossible historical candle geometry and zero-traded-volume
  bars when reconstructing touch evidence.
- Preserves an actual touch with identical duplicates and tolerates
  provisional/unfinished revisions that are already excluded by the
  existing bar-completion rules.

Only historical thesis recovery (restart and ATM state rebuilding) changes.
No entry thresholds, signal score, position size, ATR risk geometry, setup
identity schema, broker order route, or LIVE environment variables change.
Use existing focused VWAP tests, full CI and final merged-head verification.
These engineering regressions do not demonstrate higher post-cost expectancy.


## 2026-10-18: paired VWAP retest research integrity

**Scope:** SHADOW-only `run_vwap_comparison` research reporting. No trading
strategy, admission setting, premium risk geometry, broker fill, or LIVE
configuration is changed in this batch.

The existing scenario matrix already includes both
`trend_premium_rvol_1_5` and `trend_premium_vwap_retest`. This is the
correct comparison for the **incremental** prior-bar retest requirement:
comparing the ordinary `current_reference` baseline directly to retest
would simultaneously change trend, underlying-quality and independent option
volume confirmation, making the apparent retest effect unidentifiable.

Corrections and prespecified interpretation:

- Every scenario now explicitly turns off the unrelated
  `VWAP_REQUIRE_FINALIZED_SIGNAL_BAR` switch, even if inherited as enabled
  from an operator environment. Existing environment values are restored
  exactly after each replay. The original environment remains recorded as
  provenance, not as the effective isolated research configuration.
- The canonical comparison returns an explicit side-by-side,
  same-slippage contrast at **10, 25 and 50 bps per side**, showing control
  and treatment development trade counts, whether each arm has at least 30
  resolved development trades with zero unresolved exits, and the
  **treatment-minus-control net expectancy** only when evaluable.
- The descriptive chronological holdout delta is null when an arm has no
  resolved holdout trades or any unresolved holdout exits. It remains
  **retrospective and reused**: never claim untouched confirmation.
- The component runner keeps next-minute-open execution, the canonical cost
  estimator, its existing stop/target and 0.6R lifecycle proxy. **No
  dataset or profitable result is manufactured** by these code checks.

### Remaining blockers before any retest LIVE promotion

The prior-bar retest test in the existing opt-in strategy currently compares
the *preceding candle* to the *current* VWAP. A true timestamp-aligned
historical VWAP for the preceding bar must be verified before making causal
performance claims about the retest rule. The archived contract basket also
cannot establish historical ATM selection and lacks executable bid/ask/depth.
Run this corrected report on sufficiently complete real archives, then reserve
a chronologically later unseen sample and prospective recorded-feed replay.
Until then, `selected_for_live=None` and `promotion_eligible=False`.


## 2026-10-18: correct prior-bar retest VWAP look-ahead (research-only)

The opt-in `VWAP_TREND_REQUIRE_RETEST=true` branch previously compared the
**preceding completed candle** with the *current* VWAP, even though VWAP
can change between those candles. This made retest classification depend on
information unavailable at the earlier candle.

A new `IndicatorEngine.get_session_vwap_at(symbol, prior_bar_timestamp)`
supplies the preceding candle's own **completed-session typical-price
volume-weighted VWAP**. It excludes later candles, requires an exact complete
target candle and observed history starting from 09:15 IST, and returns no
reference when the historical snapshot is unavailable or ambiguous. Only
the opt-in retest uses this method; the existing current-vs-session/exchange
VWAP admission order, ordinary VWAP signals and all execution/risk controls
remain unchanged.

The retest's prior low/close use prior as-of VWAP; its current candle still
has to hold the current validated VWAP. Missing historical reference fails
closed with `trend_momentum_prior_vwap_unavailable`. The emitted candidate
records `trend_momentum_prior_session_vwap` for inspection.

**Evidence limitations:** OHLC typical-price VWAP is a completed-bar proxy,
not reconstructed exchange tick-by-tick VWAP. The previous-bar causal
comparison is engineering-correct within that model, but neither exact
historical exchange-reference parity nor enhanced after-cost NIFTY
options profitability has been established. The retest and momentum
research toggles remain **default OFF** and are not promoted to LIVE.


## 2026-10-18: require complete option-minute evidence for as-of retest VWAP

The previous causal as-of reference correctly excluded *future* option bars,
but accepted a historical sequence containing 09:15 and 09:17 without 09:16.
It could therefore treat incomplete archived turnover as a valid session VWAP.
A RED regression reproduced a numeric VWAP (150) from this gapped sequence.

The existing `IndicatorEngine.get_session_vwap_at` now requires every one-minute
timestamp between 09:15 IST and the requested completed bar, rejecting holes,
unfinished/provisional intermediates, and off-grid observations. Complete
zero-volume minute candles remain allowed, carrying continuity without adding
fictitious volume. No historical gaps are filled with synthetic prices or
assumed zero trading.

This is deliberately strict: a missing candle might represent a no-trade
minute or a data outage, but the stored OHLCV alone cannot distinguish them.
When uncertain, the opt-in retest returns
`trend_momentum_prior_vwap_unavailable`; it must not claim a valid reclaim.
Default VWAP entry rules, exits, contract choice, risk and LIVE execution are
unchanged. This integrity correction does not establish improved net expectancy.

## 2026-10-18: measurable historical retest evidence coverage

The bounded SHADOW-only `run_vwap_comparison` now reports a separate
`retest_archive_coverage` block computed once per archive (not 27 times
for the nine candidates × three slippage levels). It records observed
contract/minute opportunities from 09:30 to 14:58 IST, missing matching
spot/futures bars, weekly near-ATM proxy exclusions, untraded option
candles, and the remaining opportunities classified by the historical
VWAP proof: missing session opening, missing previous minute, untraded
previous minute, gaps since the session opening, or valid as-of
historical session VWAP. Totals, CE/PE splits and contract splits are
available with an eligible-denominator ready fraction; a zero eligible
denominator produces `null`.

All archived candles pass the existing strict normalized OHLC/date checks;
readiness is calculated causally by walking each contract's completed
09:15-to-prior-minute history exactly once. The previous candle must
have positive traded volume; completed intermediate zero-volume candles
preserve continuity. The report does **not** invent prices, retroactively
recreate missing ATM contracts, assume historical quote/depth fills, or
count unavailable contracts in the observed-bars denominator.

The diagnostic does not affect strategy signals, trade simulations,
costs, entry ranking, risk settings or deployment. A low coverage ratio
is a reason **not** to interpret simulated retest expectancy as
confirmation of historical efficacy, rather than a suggestion to
disable safety gates. This is engineering/data-quality work, not a
claim of positive after-cost returns.

## 2026-10-18: block sparse-session VWAP research rankings

A read-only check of the already completed production research job
`research-20261017-retained-five-30d-01` (2026-09-17 through 2026-10-16;
old deployed revision `a7fd5db7`) found only **2 admitted weekly near-ATM
option sessions**, 3,300 admitted option-bar evaluations and **zero resolved
VWAP trades across all nine research variants**. Neither previous prices nor
NIFTY option profitability can be established from these results; the completed
job predates the as-of VWAP and coverage corrections already merged to main.

The canonical `run_vwap_comparison` report now includes an explicit
`sample_sufficiency` preflight with **at least 10 distinct eligible sessions
(8 development, 2 chronological holdout)** required before even *descriptively*
ranking variants. This prevents many model fills on one day from satisfying
the existing 30-trade threshold while lacking cross-day variation. The
minimum-session counts are **operational conservatism, not statistically
derived power calculations**. Missing source session counts also block ranking;
resolved-trade scarcity is separately surfaced as a reason code.

The same preflight suppresses immediate-versus-retest development expectancy
deltas on sparse-session archives; paired trade counts and assumed slippage
remain visible for transparent auditing. Missing or undefined expectancy
cannot silently enter the ranking. No new archive collection, execution or
research worker is launched merely by reporting this preflight.

The report remains SHADOW-only, cannot promote LIVE settings, has no
untouched holdout, and does not recreate unobserved contract history,
executable depth or verified brokerage fills. Production configuration
and broker order paths were not changed.


## 2026-10-18: complete research environment isolation and holdout credibility

The existing bounded VWAP component comparison previously cleared only the
momentum, retest, and finalized-bar flags. If an operator had set
`VWAP_TREND_QUALITY_MAX_DISTANCE_ATR`, `VWAP_FUTURES_SLOPE_MIN_BPS`,
`VWAP_FUTURES_VOLUME_MIN_RATIO`, `VWAP_ALLOW_PENETRATION_ONLY_ENTRY`, or
`VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN` on the host, the supposed default
`current_reference` (and many named variants) silently inherited those
experimental settings. A RED test reproduced the baseline contamination.

Each arm now clears **all eight prespecified VWAP comparison controls**,
explicitly establishes the three default-off flags, and applies only its
own variant overrides; the original environment is restored after every
scenario group, even on exceptions. The raw host values remain in
`baseline_environment` for provenance, not as causal baseline assumptions.
Non-comparison LIVE flags are not changed or promoted.

The prior paired VWAP holdout report could also expose a net expectancy
difference with a single holdout trade or just two eligible sessions.
`holdout_sample_eligible` now requires the **same >=10 eligible-session
research preflight (>=8 development, >=2 holdout)** plus >=10 resolved
holdout trades in *each* arm, zero unresolved holdout exits and defined
net expectancy. Otherwise the holdout expectancy delta is `null`, while
the observed trade counts remain available. This ten-trade bound is only
a conservative **descriptive** minimum, not a power-based proof; the
holdout is still reused retrospective evidence. All comparisons remain
non-promotable, cost-modeled research rather than historical quote-fill
parity or reliable estimates of LIVE alpha.


## 2026-10-18: per-session VWAP research bottleneck attribution

The archive-level `no_vote_reasons` and `entry_funnel` aggregates previously
combined multiple trading days, hiding whether the apparent scarcity of
VWAP signals came from one anomalous session or occurred consistently.
The canonical bar-research `_scenario` now records
`strategies.VWAPPro.daily_diagnostics[YYYY-MM-DD]` for each actually
evaluated session: evaluated option bars, emitted signals, valid geometry,
next-minute intents, accepted fills, rejection reasons, and resolved versus
unresolved exits. Exits are attributed to the *entry session* rather than
the session boundary at which they are marked unresolved.

These fields are read-only diagnostics of the same simulation; the
existing aggregate metrics, canonical modeled fees, execution sequence,
historical contract admission, signal thresholds and order geometry are
unchanged. Rejections include both strategy no-votes and post-signal
simulator exclusions; the field uses the legacy `no_vote_reasons` name
for consistency. The stage totals reconcile with the corresponding
overall funnel across dates. An emitted signal is not automatically a
filled or economically profitable trade.

The old completed production research archive admitted only two
weekly near-ATM sessions and had no resolved VWAP trades. This new
daily information can help distinguish data scarcity from structural
strategy rejections **after the new research worker code is deployed**;
it is not a claim of improved backtest expectancy or live profitability.
