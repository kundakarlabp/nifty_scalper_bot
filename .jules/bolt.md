## 2024-05-18 - Optimized Instrument Insert Queries
**Learning:** Replaced a loop using `conn.execute()` with a batch `conn.executemany()` operation to resolve an N+1 query issue for SQLite inserts. Python-to-SQLite context switching during loop iteration adds unnecessary latency.
**Action:** Always accumulate rows into a tuple list and use `executemany` when inserting multiple records into SQLite. Be careful not to commit generated artifacts like `.coverage` test files.
