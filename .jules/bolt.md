## 2024-05-24 - Pandas Timestamp Truncation Overhead
**Learning:** Truncating `pd.Timestamp` to the nearest minute using `.floor('1min')` incurs significant overhead due to pandas' internal string parsing and frequency method calculations. This is particularly noticeable in high-throughput data processing paths like tick ingestion.
**Action:** Always use `.replace(second=0, microsecond=0, nanosecond=0)` instead of `.floor('1min')` or similar frequency methods for fast time truncation on pandas Timestamps in performance-critical sections.
