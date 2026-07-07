# Future candle timestamp incident

Root cause: a naive internal IST wall-clock candle could be interpreted as UTC and become a later stored candle. Valid current bars were then rejected as out-of-order.

Fix:
- localize naive market-data timestamps as Asia/Kolkata
- convert aware timestamps to Asia/Kolkata
- reject ahead-of-clock candles before store insertion
- move overlap tolerance into CandleStore

Validation:
- python -m compileall -q src tests
- python -m pytest -q tests/data/test_future_candle_guard.py tests/data/test_volume_zero_preservation.py tests/data/test_market_data_pipeline.py tests/data/test_pipeline_overlap_guard.py tests/data/test_candle_engine.py
