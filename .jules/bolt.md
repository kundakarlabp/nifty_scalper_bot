## 2024-06-27 - [Timestamp Truncation Overhead]
**Learning:** For truncating pd.Timestamp to the nearest minute in performance-critical paths, using `.floor('1min')` introduces significant string parsing and offset overhead associated with pandas' frequency methods.
**Action:** Use `.replace(second=0, microsecond=0, nanosecond=0)` instead of `.floor('1min')` to avoid this overhead.
