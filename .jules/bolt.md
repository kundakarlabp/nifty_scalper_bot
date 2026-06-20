## 2023-10-27 - Pandas Timestamp Truncation Optimization
**Learning:** In performance-critical paths (like processing incoming market ticks), using pandas frequency methods like `pd.Timestamp.floor('1min')` is surprisingly slow due to string parsing and offset calculation overhead.
**Action:** For truncating `pd.Timestamp` to the nearest minute, use the built-in `.replace(second=0, microsecond=0, nanosecond=0)` method instead, which is roughly ~14x faster.
