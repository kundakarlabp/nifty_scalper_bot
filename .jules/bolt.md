## 2025-03-03 - Optimizing pandas Timestamp truncation
**Learning:** Using `pd.Timestamp.replace(second=0, microsecond=0, nanosecond=0)` is significantly faster (up to ~30x) than `.floor("1min")` for timestamp truncation because it avoids pandas frequency parsing and offset overhead.
**Action:** Prefer `.replace()` over `.floor()` for truncating datetimes to the minute in performance-critical paths.
