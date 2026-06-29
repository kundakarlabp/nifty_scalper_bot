## 2024-06-29 - pd.Timestamp floor bottleneck

**Learning:** `pd.Timestamp.floor('1min')` is unexpectedly slow on high-frequency paths because it triggers internal string parsing and frequency logic within pandas.
**Action:** Use `.replace(second=0, microsecond=0, nanosecond=0)` which is roughly 28x faster for simple truncation.
