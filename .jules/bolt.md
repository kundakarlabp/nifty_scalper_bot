## 2024-06-24 - Pandas Timestamp truncation overhead
**Learning:** In performance-critical paths, truncating `pd.Timestamp` using `.floor('1min')` introduces significant string parsing and offset overhead.
**Action:** Always use `.replace(second=0, microsecond=0, nanosecond=0)` instead of `.floor('1min')` to truncate to the nearest minute. It is ~25x faster and avoids pandas frequency string parsing.
