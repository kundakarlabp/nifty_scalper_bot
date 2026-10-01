# Completed historical bar availability

Issue: #1480. Owner: `backtesting.replay.ReplayHarness`.

The replay ingress previously interpreted every index timestamp as availability
of the whole row. That is valid for observation/quote timestamps, but a broker
bar-start timestamp cannot expose its completed high/low/close at that instant.

The existing dataframe/file/day APIs now accept a keyword-only `bar_interval`.
For synchronized bar-start inputs, pass their actual fixed candle duration:

```python
from datetime import timedelta

result = harness.run_dataframe(frame, bar_interval=timedelta(minutes=1))
# Or: harness.run_file(path, bar_interval=timedelta(minutes=1))
```

All index timestamps shift to completion time before sorting, replay clock
advancement, historical basket lookup, quote publication and tick dispatch.
Result start/end therefore describe availability times. Original input frames
are not mutated. Omit the argument for already availability-stamped observations;
existing quote replay and golden fixtures keep their timestamp semantics.
Nonpositive or incorrectly typed intervals fail before replay side effects.
All option and underlying columns in the supplied frame must use the same
interval and timestamp convention; callers must synchronize their inputs.

TDD reproductions: the public API initially rejected the completion argument;
nonpositive intervals were then shown to dispatch instead of failing; file/day
forwarding was also reproduced before correction. Regressions cover one/five
minute intervals, aware IST timestamps, unsorted input, synchronized option and
index dispatch, result bounds, interval rejection without ticks/quotes/clock
mutation, and CSV/day propagation. Existing default clock/parity tests remain.
Focused/full façade validation and exact final-head CI are required before merge.

This prevents one specific availability error. Completed OHLC is not historical
FULL-depth data; the change does not prove fills, spread, slippage or a profitable
strategy. No live strategy, stop, target, sizing or risk gate is changed.
