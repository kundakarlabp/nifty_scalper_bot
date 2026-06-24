## 2024-05-18 - Faster pd.Timestamp truncation

**Learning:** `pd.Timestamp.floor('1min')` is unexpectedly slow in Pandas due to internal string parsing and offset validation logic. Using `pd.Timestamp.replace(second=0, microsecond=0, nanosecond=0)` achieves the identical effect and is around ~12x faster. This is particularly significant for performance-critical hot paths like tick processing and OHLC aggregations where every millisecond counts.
**Action:** When truncating Pandas timestamps to the minute in high-throughput components, use `.replace(second=0, microsecond=0, nanosecond=0)` instead of `.floor('1min')`.
