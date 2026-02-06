# VWAP Strategy Audit (In-Depth)

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
