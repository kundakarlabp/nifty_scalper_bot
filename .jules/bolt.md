## 2024-05-19 - Fast Timestamp Truncation
**Learning:** For truncating `pd.Timestamp` to the nearest minute in performance-critical paths, `ts.floor('1min')` incurs significant overhead due to parsing the frequency string, validating the offset, and calculating the delta.
**Action:** Use `ts.replace(second=0, microsecond=0, nanosecond=0)` instead to directly mutate the time components and bypass offset-logic overhead entirely, providing a significant latency reduction per tick. Be careful not to commit temporary test artifacts like `.coverage` files generated when running tests.
