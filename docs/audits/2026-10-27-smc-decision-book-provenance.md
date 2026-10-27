# 27 October 2026: bounded SMC decision-time book provenance

## Why

The 26 October second SMC CE trade had a **387-second-old original sweep** when
the manager approved entry, despite a fresh quote and two brief OrderFlow
observations. The independent five-minute entry safety fix is already live in
GitHub source. The historical journal allows deterministic admission-age
replay but *cannot* reconstruct a tradable bid/ask path for every rejected
setup, so it does not establish a counterfactual net expectancy.

## New observational evidence

`strategies/signal_identity.py::finalize_signal_observability` is the **existing
canonical single producer** of SMC trigger votes, including a valid SMC
trigger that is later vetoed by StrategyManager. For each SMC trigger vote it
now emits a companion `SMC_QUOTE_OBSERVATION` log event, with only:

- existing setup ID, option symbol, evaluation snapshot ID and quote version;
- bid and ask **only if** both are finite, positive, ordered, and the stamped
  vote quote version equals the indicator-context version;
- an optional indicator tick timestamp, and a boolean for two-sided depth
  presence (not depth levels or order data);
- a state separating `indicator_book_present`,
  `book_missing_or_incoherent`, and `quote_version_unproven`.

The event includes no access tokens, customer data, broker credentials,
order-book levels, external calls, or subscription changes. It does not
store a second quote cache, alter strategy signals, require extra REST calls
or introduce an independent trading authority. It is **not** a claim that
the bid/ask were executable or even atomic with the original tick: the
indicator context can combine fields from distinct normal processing stages.
Unknown/missing evidence never becomes a synthetic zero spread.

The existing `STRATEGY_TRIGGER_VOTE` remains unchanged, preserving identity
and admission contracts. The sample rate is **one additional small INFO event
per logged SMC trigger vote**, not per incoming market tick. Historical
journals cannot be retroactively augmented.

## Research limits and next validation

This is an observational instrumentation step only. After deployment, compare
new SMC book sample counts against logged trigger votes, unmatched versions,
invalid books, underlying sweep age and approval/rejection decisions on fresh
market sessions. Verify no increase in live event-loop lag and no sustained
production log-volume problem. A sample is not a full executable after-entry
bid/ask, bid size or end-to-end broker fill history. P&L of a rejected entry
**remains unknowable** from this data alone.

Before testing a new RANGE regime restriction or shorter OrderFlow persistence
rule, collect time-aligned bounded forward executable-side quote paths plus
timestamped underlying finalized bars with depth and broker-fill provenance
under a **separately reviewed, resource-bounded research plan**. Evaluate
setup-deduplicated net expectancy, missed true positives, risk exposure,
fees and drawdown on out-of-sample market days. Do not promote a setting based
on the sole 26 October loss.

## Release/deployment safeguard

The read-only production heartbeat on 27 October ~18:46 IST showed
`build_sha=5ee19ba`, `release_state=ci_failed`, quiet mode and responsive
admin/Streamlit checks while `main` was ahead. No action in this research
change bypasses the independent *successful main-push CI* requirement in
`deploy/lightsail_release.sh`. A PR-head green build is not release proof;
investigate the exact main-push CI run before claiming this instrumentation is
running on Lightsail.
