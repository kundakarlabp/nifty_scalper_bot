# Historical normalization correction

Issue: #1478. Owner: `data.normalizers.normalize_history_row`.

Historical OHLC must be finite and positive before downstream indicators consume it.
The previous comparison accepted NaN and positive infinity. Regression tests reproduced
that behavior before the existing normalization boundary was corrected.

Kite history's seventh candle field is optional OI. The normalizer discarded it;
regressions reproduced the loss for positional candles and mapping forms (`oi`,
`open_interest`). Valid nonnegative integral OI, including zero, is now preserved.
Missing, malformed, negative, fractional or non-finite OI is omitted without rejecting
otherwise valid prices. Existing timestamps, source, volume and valid OHLC behavior
remain unchanged. No new adapter, dependency, signal gate or trading path was added.

Validation: targeted regressions were observed failing before the fixes and passing
afterward; repository façade checks and final-head CI must pass before merge.
These are data correctness fixes, not evidence of improved strategy profitability.
Real intraday optimization still requires synchronized option/context history and
execution evidence; daily exchange files cannot establish intraday fill quality.
