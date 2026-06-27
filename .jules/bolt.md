
## 2024-05-18 - Pandas Timestamp Truncation Overhead
**Learning:** Using `pd.Timestamp.floor('1min')` is unexpectedly slow in performance-critical data pipelines because pandas invokes string parsing and frequency offsets under the hood. For simple truncation to the minute boundary, calling `.replace(second=0, microsecond=0, nanosecond=0)` is roughly 28x faster.
**Action:** When truncating `pd.Timestamp` to basic time boundaries in hot paths (like tick ingestion), use explicit `.replace(...)` instead of the more idiomatic but slower pandas frequency methods (`.floor`, `.ceil`, `.round`).
