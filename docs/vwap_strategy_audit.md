# VWAP Strategy Audit (In-Depth)

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
Historical-versus-current VWAP selection remains a separate open concern.
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
| P1 | `vwap_pro._recover_thesis_anchor_from_history` compares historical OHLC with the current VWAP and returns an unnormalized timestamp string. | Audit concern, not a proven production incident. Test moving-VWAP recovery and equivalent timestamp representations before changing state reconstruction. | Next focused strategy correction, after the P0 merge. |
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
