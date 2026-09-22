# Signal Pipeline Contract Audit — 2026-09-22

## Scope

This audit follows the 2026-09-21 LIVE session review of the NIFTY directional-options path:

market data -> underlying direction -> SMC/VWAPPro/ORBPro/OrderFlow -> StrategyManager -> Runner final quality -> risk/capital -> OrderManager.

The objective is not to increase trade frequency. It is to remove contradictions and dead code while preserving the existing fail-closed risk and quality contract.

## Production evidence that triggered the review

The 2026-09-21 logs showed long periods with no trigger-capable strategy vote, dominated by legitimate strategy reasons such as:

- VWAPPro: distance_outside_band and, before #1285, restart-related vwap_thesis_not_armed.
- SMC: underlying_no_liquidity_sweep, reclaim/confirmation waiting.
- ORBPro: orb_entry_window_expired after its intended session window.
- OrderFlow: context-only by design.
- No material regime or capital blocks during the reviewed interval.

A later VWAP candidate reached StrategyManager qualification but was rejected at Runner final quality with alpha_below_threshold. That rejection is intentionally retained: Runner is the final score owner and VWAP's independent alpha must not be rescued solely by execution quality or overlapping context evidence.

## Evidence review

### Order flow

Short-horizon order-flow imbalance is informative, but its value is horizon-, liquidity-, and market-dependent.

- Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events", Journal of Financial Econometrics (2014), DOI 10.1093/jjfinec/nbt003: short-horizon price changes are strongly related to order-flow imbalance and market depth.
- Tripathi, Dixit & Vipul, "Information content of order imbalance in an order-driven market: Indian Evidence", Finance Research Letters (2021), DOI 10.1016/j.frl.2020.101863: NSE order imbalance predicted short-term returns most strongly over the first minutes and the effect decayed materially at longer horizons.
- Lee, Ryu & Yang, "Does vega-neutral options trading contain information?", Journal of Empirical Finance (2021), DOI 10.1016/j.jempfin.2021.04.003: aggregate option imbalance can lose incremental information after controlling for futures flow, while selected option-flow decompositions retain information.
- Sensoy & Omole, "Information content of order imbalance in the index options market", International Review of Economics & Finance (2022), DOI 10.1016/j.iref.2021.11.006: index-option order imbalance contains information, with hedging and cross-market channels relevant to interpretation.
- Yamamoto, "Intraday technical analysis of individual stocks on the Tokyo Stock Exchange", Journal of Banking & Finance (2012), DOI 10.1016/j.jbankfin.2012.07.006: apparent short-horizon predictability does not automatically survive realistic execution and data-snooping considerations.

Engineering implication: OrderFlow remains context-only. It may confirm or veto a trigger, but it is not added as a second copy of directional alpha when underlying/futures information already contributes to direction.

### VWAP

VWAP is first and foremost a traded-volume execution/price benchmark. A price being above or below VWAP is not, by itself, sufficient evidence of directional alpha. The bot therefore retains its premium event/reclaim/continuation evidence, distance control, underlying-direction alignment, and Runner final-quality floor.

## Canonical ownership after this change

### StrategyManager

StrategyManager owns strategy evaluation, trigger/context arbitration and candidate qualification. Its positive terminal event is STRATEGY_CANDIDATE_QUALIFIED, not SIGNAL_APPROVED.

The existing is_approved metadata flag is retained as a compatibility field because it is used by current StrategyManager/live-safety control flow. It means "qualified to leave StrategyManager", not "approved for a live order".

### Runner

Runner remains the sole final numeric quality/alpha owner. It uses score_signal_metadata(), which delegates to the existing score_signal_quality() model.

For VWAPPro only, the strategy component remains independent_setup_score. This deliberately excludes underlying direction, futures slope and futures volume so those evidence families are not double-counted.

Runner emits SIGNAL_APPROVED only after the canonical final-quality gate returns allowed=True. This event still does not mean an order was submitted or filled; execution, risk, order-submission and fill events remain separate.

### Counters

Two different quantities are now explicit:

- candidate_generated_count / candidate_generated: StrategyManager produced a candidate.
- approved_candidate_count / approved_candidates: Runner final quality accepted the candidate.

Neither counter is an order/fill counter.

## Corrections

1. Added score_signal_metadata() as the canonical adapter from Runner-ready metadata to the existing final quality model. No alternate scoring engine or threshold was added.
2. Preserved the VWAP independent-alpha floor; OrderFlow context is not added to VWAP alpha.
3. Renamed StrategyManager positive telemetry from SIGNAL_APPROVED to STRATEGY_CANDIDATE_QUALIFIED. Runner now owns SIGNAL_APPROVED after final quality.
4. Split generated-candidate and final-quality-approved counters.
5. Moved same-contract, same-session VWAP history recovery before the distance_outside_band early return. The distance gate itself is unchanged. This restores restart state even while the option is temporarily overextended.
6. Removed the LTP-vs-close pseudo-fallback that only appended a reason and never changed CE/PE scores. Tick movement remains non-authoritative and cannot manufacture underlying direction.
7. Fixed StrategyManager exit diagnostics at the source: score is read from canonical score metadata rather than order quantity.
8. Removed the obsolete strategy_exit_score_diagnostics runtime monkey patch and its installer.

## Explicit non-changes

The following controls are unchanged:

- per-trade and daily risk configuration;
- live spread/quote-age requirements;
- VWAP trigger and Runner alpha thresholds;
- distance_outside_band;
- SMC sweep/reclaim/BOS/retest requirements;
- ORB session window;
- OrderFlow context-only role;
- underlying spot/futures direction authority;
- hard context vetoes;
- candidate selection, capital checks, risk sizing and order routing.

## TDD coverage

Regression tests establish:

- a strong RANGE VWAP + OrderFlow candidate can reach Manager qualification;
- the same candidate still fails closed when independent VWAP alpha is below Runner's unchanged floor;
- a sufficiently strong independent-alpha candidate can pass Runner quality without threshold relaxation;
- LTP-close movement alone cannot manufacture underlying direction or claim a nonexistent fallback;
- VWAP restart state is recovered even when current premium is outside the entry distance band;
- Manager qualification and Runner approval telemetry are distinct;
- candidate and final-quality-approved counters measure different stages;
- StrategyManager diagnostics no longer use quantity as score;
- core startup no longer installs the removed diagnostics monkey patch.

## Deferred work

The repository contains other historical runtime-hardening adapters/import hooks. They are not causal to the reviewed 2026-09-21 signal blockage and are intentionally not rewritten in this focused change. Broadly replacing them here would increase regression surface without evidence of benefit. They should be canonicalized separately, one owner at a time, with dedicated parity tests.

## Validation requirements

Before merge:

1. focused regression tests pass;
2. full repository tests pass;
3. Ruff/Black/mypy/compile quality checks pass;
4. deterministic broker-free E2E passes;
5. PR diff is reviewed for threshold/risk changes;
6. main is merged only after all required checks are green.

Post-deploy validation should verify build SHA, readiness/arming, StrategyManager candidate counts vs Runner approval counts, final-quality rejection reasons, and actual order/fill events independently.
