# 27 October 2026 — VWAPPro live context-confidence counterfactual

**Decision: NO LIVE THRESHOLD CHANGE.** This is a read-only evidence and diagnostics correction. A lower confidence floor remains an unapproved experiment until a matched chronological executable-quote replay has demonstrated an improvement in net performance and acceptable drawdown.

## Reproducible code finding

- `src/nifty_scalper_bot/core/strategy_manager.py::_derive_context_direction` awards 0.80 for structurally agreed underlying direction, adds only 0.05 for supporting short-term movement, and subtracts 0.05 on disagreement. Ordinary conclusive source confidence therefore tops out at 0.85. This confidence is explicitly provenance telemetry in the direction engine, not proof of trade profitability.
- `VWAPProStrategy.__init__` has `VWAP_EARLY_TREND_MIN_CONTEXT_CONF` default 0.90 and freshness limit `VWAP_EARLY_TREND_MAX_CONTEXT_AGE` default two seconds. Unlike ordinary VWAP entries, premium trend-momentum/neutral-slope/extension candidates require `strong_fresh_trend_context`; the 0.90 default conflicts with the ordinary conclusive direction producer.
- A valid session, side, fresh bar, trend regime, completed candle, volume, nonstale executable quote, spread, risk geometry and final admission remain separate requirements. The 0.80 unit fixture is **not** an actual trade, a full strategy candidate count, or a fill.
- The running process may override defaults via its environment. The new `VWAP_EARLY_TREND_CONTEXT_GATE_CONFIG` initialization event records effective min confidence, age, and mode switches after deployment without revealing credentials. Do not call the default the confirmed live override until that event is observed.

## Archived live evidence, read 27 October 2026

Supabase archived journald, `public.nifty_daily_market_logs`, covering 24 sessions from 24 September to 27 October 2026, is operational telemetry, **not** historical executable tick-by-tick quotes. Across October 21, 22, 23, 26 and 27, the logged `DIRECTION_CONTEXT_RESOLVED` events were:

| Logged confidence | Events |
| --- | ---: |
| 0.50 | 5 |
| 0.80 | 2,222 |
| 0.90 or higher | 0 |

These are **repeated direction reports**, not independent trade setups; never treat them as 2,222 missing entries. For example, the 27 October ~13:19–13:22 IST `VWAPPro` `trend_momentum_underlying_quality_unconfirmed` reason occurred while PE direction was reported at 0.80. Other mutually constraining no-votes included VWAP distance, option candle no trades, missing SMC sweep, and ORB entry window expiry. Merely crossing the context-confidence floor does not establish a valid, fillable candidate.

For October 1–23, the `nifty_trade_ledger` records **eight completed VWAPPro trades**, aggregate **+₹218.00 net** after **₹514.97 estimated costs**. Three recent VWAP entries include confidence 0.80 in their entry snapshots and were admitted on an existing ordinary `single_trigger_context_confirmed` path; these observed successful/unsuccessful fills must **not** be re-labeled as counterfactual early-trend fills. All historical rows included 321 ledger entries, of which only 105 had entry and exit prices and cost estimates. The event stream did not contain a complete matched bid/ask/indicator history for rejected candidate timestamps.

## Counterfactual comparison, what is actually demonstrated

| Arm | Context-only unit-fixture verdict at 0.80 | Realized historical P&L attributable to this arm |
| --- | --- | --- |
| Default floor 0.90 | Reject for trend-momentum quality | Not computable |
| Proposed floor 0.80 | Can advance on a correctly completed synthetic trend/premium bar, provided all other protections pass | Not computable |

The focused tests verify (1) threshold effect on exactly one controlled event and (2) that regime, age and completed-bar gates still reject under 0.80. **This is a code-path test, not a realistic backtest or transaction-cost study.**

## Promotion gate — fail closed until verifiable

1. Obtain reproducible session-scoped option-contract identities, contemporaneous NIFTY spot/futures context, all completed option candles, original exchange/session VWAP, ATR, volume, book bid/ask plus depth and quote timestamps for candidate moments, including **rejected** setups. Preserve raw source and observed-at timestamps.
2. Replay 0.90 and 0.80 on identical time-ordered event streams, with separate strategy instances, the same causal history, identical ATM selection, allocation, position contention, trailing exits, underlying direction provenance and no look-ahead.
3. Model **executable** BUY ask and SELL bid, spread, slippage, partial fills, brokerage, statutory charges, latency, protective-order realism and exits. No candle-close proxy presented as a real fill.
4. Compare incremental unique setups, admissions, fills, net P&L, profit factor, costs, false-entry/stop rate, maximum intraday drawdown and downside by market regime; provide out-of-sample walk-forward sessions and sensitivity to higher costs.
5. Promote only after a materially positive after-cost result and no unacceptable deterioration in drawdown or safety; otherwise retain the 0.90 production setting or redesign the provenance-based requirement as a separately validated strategy change.

The existing live configuration, buy/exit authorization, stop-loss and release safety gates remain unchanged by this audit.
