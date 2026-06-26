## 2024-05-15 - pandas.Timestamp frequency method overhead
**Learning:** Using `pd.Timestamp.floor('1min')` is surprisingly expensive in hot loops (like tick processing) because pandas frequency parsing and offset logic involves significant overhead and string evaluation.
**Action:** Replace `ts.floor('1min')` with `.replace(second=0, microsecond=0, nanosecond=0)` to truncate timestamps to the nearest minute. This provides identical functionality for 1-minute bucketing but avoids the pandas offset machinery entirely.
