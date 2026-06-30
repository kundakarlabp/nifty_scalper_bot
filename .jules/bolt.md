## 2024-06-30 - Membership Tests with Constant Literals
**Learning:** Found several places where list literals are used with the `in` operator (e.g., `x in ["A", "B"]`).
**Action:** Replace `in [...]` with `in {...}` for faster membership tests, since set literals are pre-compiled to `frozenset` by Python 3.11+, making lookups O(1).
## 2024-06-30 - pd.Timestamp performance optimization
**Learning:** pandas `.floor('1min')` is ~10x slower than `.replace(second=0, microsecond=0, nanosecond=0)` because of string parsing and offset overhead.
**Action:** Replace `.floor('1min')` with `.replace(second=0, microsecond=0, nanosecond=0)` in performance critical paths like pipeline tick processing to speed up execution by nearly 10x.
