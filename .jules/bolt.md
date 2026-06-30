## 2024-06-30 - Batch SQLite Inserts

**Learning:** Using `conn.execute` within a loop for massive numbers of rows (like instrument data dumps) causes significant N+1 query overhead in SQLite, severely bottlenecking data load processes.
**Action:** Always accumulate parsed/extracted parameters into a list of tuples first, and use a single `conn.executemany` outside the loop to execute the bulk insert/update operation, resulting in significant speedup.
