# Market data completion validation

Run these checks locally before increasing live size:

```bash
python -m compileall -q src tests
python -m pytest -q \
  tests/streaming/test_market_data_hardening.py \
  tests/data/test_market_data_hardening.py
python -m pytest -q
```

Market-hours paper/live-small validation checklist:

1. Confirm one `FIRST_TICK_RECEIVED` event per WS session and no duplicate-enqueue warnings.
2. Confirm `MDM_CANDLE_FLUSH_TASK_STARTED` after the MDM event loop is wired.
3. Watch for `MDM_CANDLE_CLOCK_FLUSHED` after idle one-minute buckets.
4. Confirm stale/synthetic WS LTP does not arm hard trading readiness.
5. Confirm fallback worker log appears only before the main asyncio loop is wired.
6. Keep live order size minimal until one complete market session passes without feed-health degradation.
