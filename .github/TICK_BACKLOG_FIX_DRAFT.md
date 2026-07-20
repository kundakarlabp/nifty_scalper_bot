# Draft implementation marker

This temporary file exists only to allow creation of a focused draft pull request for the required-symbol tick-backlog correction.

It must be deleted in the implementation commit before the pull request is marked ready for review.

Implementation scope:

- `src/nifty_scalper_bot/data/market_data_manager.py`
- `tests/data/test_mdm_tick_coalescing.py`

The live safety gate `live_latest_closed_bar_stale` must remain fail-closed.
