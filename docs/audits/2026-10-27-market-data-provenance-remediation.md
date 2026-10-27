# 27 October 2026 NIFTY market-data provenance remediation

## Evidence and scope

The finalized Supabase log archive held 16,788 lines for 27 October. Six trading
engine starts occurred. Among 366 history-sync attempts, 84 failed
`source_history_short`, all within 9–29 seconds of a startup; 282 succeeded.
All 48 sampled history-consistency snapshots agreed by count. These are
startup/hydration observations, not proof of 84 sustained bad data episodes.

NIFTY spot and futures histories were typically approximately 399 and
390 bars at startup, respectively. Selected-option history was about
60–105 bars after warmup. `option_candle_no_trades` appeared in 79
throttled manager records (27 October) versus four (26 October); these
are **not** independent trade opportunities or proof of faulty broker volume.

## Confirmed code-level issues

1. A prior `signal_generator.StrategyManager._augment_futures_metrics` path
   replaced missing futures VWAP with futures LTP, then applied a fixed
   basis multiplier `0.998` and published the result as both
   `nifty_index_vwap` and `futures_vwap`. A synthetic index VWAP is not
   actual exchange-volume-weighted price; NIFTY spot is not a traded
   volume instrument. Futures and index values must not be conflated.
2. Existing VWAP option-volume rejection logs read `is_complete` and
   `is_provisional` from an indicator dictionary that typically does not
   expose them. This obscured whether the rejected option bar had
   finalized and whether the zero value was meaningful.
3. A complete history fetch regularly returned mostly previously ingested
   rows. Duplicate imports are reconciled, but count equality alone cannot
   prove volume accuracy or source timestamp validity.

## Implemented provenance correction

- `nifty_index_ltp` is spot-quote LTP only (a cached fallback is labelled
  `spot_quote_ltp_cached`); no basis-adjusted future is written there.
- `nifty_index_vwap` is explicitly unavailable, with
  `nifty_index_vwap_source=untraded_index_no_vwap`.
  A quote-supplied index average can be inspected separately as
  `nifty_index_quote_average_price`, never an index VWAP.
- `futures_ltp`/`nifty_fut_ltp` represent actual active-futures quote LTP.
  `futures_vwap`/`nifty_fut_vwap` are set only from a finite, positive
  futures quote VWAP or average-price value; a missing VWAP remains missing.
  The live futures context snapshot can separately supply its verified
  candle VWAP. No fixed 0.998 basis is used.
- VWAPPro's **existing no-vote gate stays unchanged**. On a bounded
  per-symbol `option_candle_no_trades` diagnostic, the code now inspects
  the actual latest indicator-history bar (`count=1, field=bars`) and
  the already-fetched cached option quote. It logs bar timestamp, finality,
  history/indicator volumes, quote delta and cumulative volumes,
  volume-transition state and evidence classification. This is
  diagnostic, not a synthetic trade or an admission override.

## Still unverified

- Whether broker cumulative volume changes correspond to the exact
  finalized minute bars, including delayed/coalesced ticks.
- Whether zero-volume rejections are genuinely zero-trade minutes, a
  provisional tail, or a data-ingestion discrepancy; the 27 October
  archive was produced *before* the more informative diagnostics.
- Comparison of exchange historical bars against exact live finalized
  OHLCV for matching symbol/time; historical bar-count parity does not
  prove content equality.
- Whether an entry relaxation has positive after-cost expectancy.
- Operating-system or deployment causes of the six startup cycles.

## Acceptance / safety constraints

Run unit tests, unchanged strategy/no-vote path tests, normal shards,
changed-code quality and broker-free E2E CI before merging. New
trade orders, changing risk, relaxing VWAP volume requirements, exposing
secrets or invoking broker trade actions are explicitly out of scope.
Verify post-merge main CI and production deployed SHA via Supabase heartbeat.
