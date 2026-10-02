# On-demand historical collection and production replay

The admin dashboard has two actions: **Start research** collects and caches broker
candles, runs component research and attempts recorded-feed replay; **Replay
recorded sessions** runs only the captured production feed, without broker login.
Requests cover 1–90 prior calendar days. Jobs share a single-flight lock. Replay
sessions have a ten-minute timeout and a total 25-minute replay budget; remaining
sessions are reported as deferred. No replay worker receives operator credentials.

The live market-data owner records delivered normalized observations under the
persistent data directory (`replay_archive`). Capture is enabled by default;
`REPLAY_CAPTURE_ENABLED=false` disables it. Snapshots preserve contract catalog,
basket identity, finalized warm-up candles, effective decision settings, runner
configuration, initial balance and release identity. Tick records preserve depth,
OI, source and timestamps as delivered. Writer queue and session size are bounded.
Capture failure never stops trading, but invalidates replay evidence. `/status`
exposes pending, written, dropped and failed capture counts. Archive files contain
market data and decision settings, not broker credentials.

Historical collection includes spot, futures, selected options and neighboring
strikes across the nearest two available expiries. A persistent checksum-validated
cache reuses identical requests across jobs. Requests retry transient errors and
report missing candles explicitly. Instrument identity is retained with candles.
This does not recover deleted expired option history or historical quote depth.

Recorded replay initializes the actual application composition and uses its market
normalization, candle, indicator, strategy, readiness, risk, order and bracket
owners. The simulation uses the live strategy mode policy. A virtual clock advances
only after queued decision work drains. Socket access is blocked. Captured baskets
are installed without fetching new contracts. Missing warm-up stays missing and
continues to block readiness. LIMIT execution respects observed top-level size,
slippage feasibility and cumulative fees. Reports include orders, modeled account
equity, fees, drawdown, open positions, capture SHA and decision diagnostics.

These reports are research candidates, not proof of full live equivalence. Prior
cooldown/risk/decision state is not restored; sessions starting with positions are
rejected. Basket choices are replayed rather than independently selected. Background
recovery and scheduling differ from live operation. Fills remain modeled. Missing
sessions, corrupt or truncated recordings, queue drops and unavailable prerequisites
are reported rather than replaced with invented quotes. Full historical fidelity
starts with data captured after this release; older component tests retain their
separate scope. Changing strategy parameters and demonstrating profitability requires
chronological holdout evaluation after valid session data is available.
