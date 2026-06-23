
## 2024-06-23 - Pandas Timestamp Trucation Bottleneck
**Learning:** Using `pd.Timestamp.floor("1min")` is a significant performance bottleneck due to its timezone and frequency-offset validation logic. In hot loops like the scalper bot's tick processing, it can degrade throughput. Using `.replace(second=0, microsecond=0, nanosecond=0)` achieves the same minute truncation roughly 27x faster.
**Action:** Always prefer `.replace()` for timestamp truncation in performance-critical paths over pandas frequency methods.
