## 2023-10-25 - Timestamp truncation performance in critical path
**Learning:** Pandas `pd.Timestamp.floor('1min')` is unexpectedly slow because it involves string parsing for the frequency offset. It can become a bottleneck in high-frequency event loops like tick processing.
**Action:** Use `.replace(second=0, microsecond=0, nanosecond=0)` instead of `.floor()` when truncating timestamps in performance-critical code paths, as it avoids frequency string parsing and is an order of magnitude faster.
