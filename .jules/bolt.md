## 2026-06-20 - [Timestamp Parsing Optimization]
**Learning:** Using `pd.Timestamp.floor('1min')` involves a hidden performance penalty because Pandas must parse the string frequency offset ('1min') before applying the calculation. In high-frequency ingestion pipelines, this string parsing can compound to a measurable bottleneck.
**Action:** Replaced `.floor('1min')` with `.replace(second=0, microsecond=0, nanosecond=0)` to avoid frequency string parsing, drastically speeding up time truncation while achieving the exact same result.
