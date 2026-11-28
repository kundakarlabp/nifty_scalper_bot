# Four-phase NIFTY strategy evidence audit — 12 November 2026

**Disposition: HOLD new LIVE promotion.** Research only; no change to StrategyManager, risk gates, broker execution, systemd, credentials or production config. This document records independently inspectable source limitations and explicit falsification criteria, not a promise of profitability.

## Phase 1 — Historical coverage and execution-cost fitness

Source reports:
- `breeze_2yr_collection_audit_20261109.md`
- `breeze_offline_archive_validation_20261109.md`
- `execution_evidence_upgrade_20261109.md`

Existing immutable acquisition reports enumerate 25 encrypted windows (2024-11-06 to 2026-11-05); 6 collector-complete, 19 partial. Row totals: index 184,786; identified futures 536,695; selected CE/PE 733,386; total 1,454,867 one-minute bars. All 25 archives were reportedly checksum-verifiable, but this report has not independently reopened or decrypted the private source archives. Repair of 158 contract-days recovered zero additional candles (139 matching replies, 19 provider failures).

**NOT execution grade:** the selected CE/PE archive is not a historical options chain; it lacks contemporaneous point-in-time strike eligibility, bid/ask, five-level depth, measured arrival/exit slippage, broker queue/fills, time-aligned implied volatility and independently verified bar-time conventions. All 34 empty futures contract-days, 19 persistent failures and 105 normal-session minute-gap series remain qualified. The 2025-02-01 special Budget session is absent; 2025-10-21 Muhurat index/bar boundary is unresolved. Treat missing source evidence as `UNRESOLVED`, never as no signal or zero cost.

Execution-cost rules:
- Fee models and lot size must be indexed by actual execution date, instrument expiry, venue and actual broker; the research `historical_option_fee` requires an explicitly verified exchange fee rate instead of back-applying today's rate.
- NSE options STT on a premium sell is 0.10% before 2026-04-01, 0.15% thereafter; settlement/exercise charges need a distinct documented policy. Brokerage, GST, exchange, SEBI, stamp and slippage must be per executed leg, including exits and failed/partial basket recovery. Exchange and broker contract notes override estimates.
- The NIFTY market lot changed over the study period; never apply today's 65 units to all trades.
- The canonical `ReplayHarness(strict_execution_evidence=True)` must remain fail-closed on historical OHLCV, not infer executable prices from candles.

Official rate and contract sources:
- https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax
- https://www.nseindia.com/static/products-services/equity-derivatives-contract-specifications
- https://zerodha.com/charges/
- https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf

## Phase 1B — Reconcile no-trade evidence

Obtain the latest completed market-session snapshot before making any claim about VWAP, ORB, SMC or candidate rejection rates. Require one row per decision/event with:
`session_date, selection_epoch, candle_version, selected_symbol/token, evaluation_time, manager_calls, last_evaluation_age_ms, strategy, event/setup_id, stage, reason_code, context_source, quote_timestamp/source, spread, cost, risk, broker_preflight`.
Deduplicate repeated per-minute no-votes by stable event/setup ID and selected basket epoch. Separate the denominators:
1. expected market evaluation opportunities;
2. complete causal context opportunities;
3. triggered candidate events;
4. approved strategy decisions;
5. market/contract/risk-qualified trades;
6. broker-executed entries and exits.

The latest off-hours `quiet`/AUTO observation is a scheduled state, *not* a trading rejection. If the read-only live endpoint is unavailable, record `LATEST_LIVE_REJECTIONS_UNVERIFIED` and reconcile later from the existing canonical journal rather than inventing rejection counts.

Do not relax spread, liquidity, capital, instrument expiry/identity, stale-quote, stop, broker preflight, position reconciliation, session or duplicate-intent gates.

## Phase 2 — Matched directional event research

Research candidates:
- VWAPPro continuation reference vs prespecified retest/rebreak-only ablation.
- ORBPro baseline vs momentum-only / retest-only timing and opening-range sensitivity.
- Independent `FailedBreakoutResearch` (sweep/reclaim and later completed confirmation) versus the existing SMC, with a common underlying-futures event universe.

Use *one canonical market-event table*, immutable before outcome evaluation, keyed by IST session, underlying futures event timestamp, boundary/level, side, regime as-of, snapshot-version and event ID. For every event keep candidate/no-vote/rejected/no-fill observations from all arms, including events for which SMC has no setup. Do not compare cherry-picked signal subsets as if event universes were matched.

Price the SAME point-in-time historically eligible expiry/option strike for every comparable arm; when a contract is unidentifiable or depth unavailable mark both potential outcomes as non-executable, not modeled wins. A signal at bar close is eligible only for subsequent market observation. Freeze exits, sizing, daily limits and slip/fee policy across arms. Record ambiguous intrabar stop/target ordering explicitly and include adverse scenario.

Report independently for each chronological development/validation/untouched-holdout cohort: event counts, context coverage, distinct qualified signals, rejection and no-fill stages, net expectancy per trade *and per eligible event*, costs, payoff distribution, max equity drawdown, tail loss/expected shortfall, regime breakdown and uncertainty under session-blocked resampling. Track all attempted parameter choices and reject selection driven by one fortunate day. No historical-bid/ask data means a **causal signal comparison** at most, not a broker-executable profitability claim.

Existing relevant code: `backtesting/directional_hypotheses.py`, `scripts/research_orb_multiyear.py`, `strategies/elite_strategies/{vwap_pro,orb_pro}.py`, `docs/research/2026-10-30-failed-breakout-reversal.md`.

## Phase 3 — Compare option structures on the SAME eligible directional event

First compare:
- A single long CE/PE, crossed at observed ask, subsequently sold to observed bid.
- Bull-call debit spread or bear-put debit spread, same expiry, same lot and synchronized identified book; buy nearer/in-the-money leg at ask, sell more out-of-money wing at bid. No opposite-sign or free-money debit anomalies.
- Per-order fees for all legs on both entry and exit; inventory and broker-margin proofs for selling the vertical wing; incomplete baskets are **unresolved exposure**, not a completed strategy.
- Identical directional event, signal timestamp, risk limit and hold policy. Compare net returns, after-cost capital efficiency, tail loss and rejected/unaffordable opportunities.

The research-only `assess_directional_debit_spread` computes strictly synchronized *entry geometry* and the single-long comparison from observed FULL/archived-FULL quotes; it DOES NOT claim actual fills, performance, future win probability, certified margin, or realized downside. Its `live_eligible` is permanently false. A separate observed exit ledger is required to compare realized outcomes.

Only after this is available investigate short credit spreads and four-leg condors on a prespecified, prospective volatility regime. Require an IV forecast available before the observation, full option-chain coverage and each basket leg's executable book. Condor screens are not approved position execution; group stop, margin reserve, wing-first recovery and broker reconciliation must be independently tested. Existing P4/P5 research and paper controls remain subject to `2026-10-31-p45-prospective-evidence-gate.md`.

## Phase 4 — Regime/portfolio/paper policy

Use solely as-of features: opening-range width/ATR, prior-session volatility, completed-bar trend efficiency, realized-vs-implied forecast gap, expiry distance and quote liquidity. Do not label an entire day using what happened after an entry. Start from **no-trade** and existing directional-only baseline; evaluate regime assignment versus a non-regime benchmark on untouched chronology.

Retain the authoritative per-trade cost-inclusive loss cap, independent daily cap, capital, gross premiums, combined option Greeks, concentration by expiry/side, correlated positions, exit capacity, margin reserves, terminal-order reconciliation and fail-closed outage behavior. Portfolio-loss stressed reserves include all leg execution costs and short-leg rescue/recovery, not only terminal payoff geometry. If positions conflict or evidence is missing, abstain.

The existing `backtesting/volatility_paper_execution.py` and `execution/position_manager.py` linked portfolios do not certify that LIVE multi-leg entry is safe. PAPER/SHADOW simulations must record every intended child, partial, reject, cancel, timeout, restart, missing wing, forced flatten and independent broker-position reconciliation. Require both:
1. stable, positive held-out **net** expectancy with realistic adverse cost scenarios and meaningful opportunity denominator;
2. prospective sessions with known book quality, isolated paper outcomes and verified exits/recovery.
Any strategy failing either criterion remains `RESEARCH_CANDIDATE`.

## What is explicitly NOT completed

- Fresh production-session per-event rejection analysis (read-only live endpoint/complete decision journal not supplied for this report).
- Full-chain, five-level historical option snapshot reconstruction (not derivable from the archived candles).
- Matched executable VWAP/ORB/reversal trades with validated cost-inclusive P&L and untouched holdout (insufficient source books).
- Executed longitudinal debit-spread, credit-spread or iron-condor returns (missing verified entry/exit fills and margin).
- Portfolio OOS alpha or promotion of strategies to LIVE.

**Recommended decision:** keep LIVE unchanged; collect or license missing point-in-time books and record prospective PAPER/SHADOW; run the event-matched study only after the evidence gate passes. `positive_net_alpha_demonstrated=false`, `live_promotion_eligible=false`.
