# ORB retained-history replay: separate session staging from opening-range readiness

Date: 2026-10-26 (IST). Status: engineering/research infrastructure; **not live-equivalent** and **not promotion eligible**.

## Diagnosed defect

The retained Kite-history inventory already requires date-causal, independently
observed spot, NIFTY future and weekly-eligible near-ATM CE/PE contracts, verified
instrument/token identity, and contiguous shared post-09:30 warmup minutes.
That generic condition is **insufficient** for a 15-minute ORB, whose opening
range is built exclusively from the underlying NIFTY future's 09:15 through
09:29 IST completed one-minute candles. A session could be staged from 09:30
even when some or all futures opening-range minutes were absent.

Repeated per-bar no-trade counts cannot be treated as unique failed ORB setups.
Historical missingness cannot be reclassified as weak alpha or no signal.

## Canonical correction

Research owner: `scripts/archive_kite_history.py::historical_cache_inventory`.

- After a session's **observed** contract pair is stageable, count missing
  futures timestamps for each minute 09:15–09:29 IST inclusive.
- Report `orb_opening_range_ready_staged_session_dates` (zero missing),
  `orb_opening_range_incomplete_staged_session_dates` (one or more missing),
  and `orb_opening_range_missing_futures_minutes` keyed by date.
- Preserve all generic stageable sessions and their factual coverage: a
  separately incomplete ORB opening range must not erase possible SMC/VWAP
  observations, but such a session is not complete ORB evidence.
- Propagate missing-minute count and `orb_opening_range_complete` into
  `run_research_job.py`'s per-session summary and persisted component report.
  Unknown missingness is **not** considered complete (fail closed).
- Keep `promotion_eligible=false`, `historical_active_universe_proven=false`,
  `live_basket_cap_unchanged=true`, and existing baseline computations.
- Do **not** infer live-equivalence from complete opening futures timestamps:
  full session continuity, correctly recorded market data, historical basket
  identity, executable option bid/ask and realistic order fills remain unproved.

## Research protocol before discussing LIVE thresholds

1. Prefer the existing production recorded-feed SSOT
   `src/nifty_scalper_bot/storage/replay_archive.py` and
   `backtesting/runtime_research.py`; do not construct a competing collector.
2. For historical sessions, accept only verified historical futures expiry,
   instrument token, weekly options expiry/strike at the *original event time*,
   contiguous required bars, and recorded depth/quotes; reject unsupported
   data rather than filling gaps with monthly/front-week substitutes.
3. Record eligible dates/symbols, missing opening range per date, number of
   unique setups and passage through strategy, admission, cash/stop affordability
   and broker (or labeled proxy) exit. Deduplicate quote evaluations by setup ID.
4. Compare the current setup with **one prespecified change at a time**:
   VWAP retest/slope, OrderFlow reversal holdoff or ORB RANGE-breakout gate.
   Preserve 10/25/50 bps and realistic fees, slippage, stop-first ambiguity,
   unresolved orders and cost-inclusive net results. Development/validation/
   untouched holdout must be chronological.
5. No out-of-sample expectancy claim and **no automatic LIVE promotion** while
   #1663 / #1686 coverage gaps persist. Recorded-feed option history cannot
   retroactively reconstruct expired symbols not originally captured.

Related: #1663, #1686, #1659; supersedes stale changes proposed in #1740
without changing any LIVE broker, risk, entry, exit, or protective order setting.
