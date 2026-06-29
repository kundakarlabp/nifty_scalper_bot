
## 2024-06-29 - Optimize pd.Timestamp truncation
**Learning:** For truncating `pd.Timestamp` to the nearest minute in performance-critical paths, `.replace(second=0, microsecond=0, nanosecond=0)` is significantly faster (around ~27x faster) than `.floor('1min')`. This is because `.replace` avoids the significant string parsing and offset calculation overhead associated with pandas' frequency methods.
**Action:** Replace `.floor('1min')` with `.replace(second=0, microsecond=0, nanosecond=0)` in hot paths, specifically in data processing pipelines like tick to candle aggregations.
