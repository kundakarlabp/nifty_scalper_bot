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


## Automatic read-only quote-fidelity preflight

Every invocation of `scripts/run_runtime_replay.py` now writes
`capture-fidelity.json` beside the existing modeled replay report **before**
running the unchanged production-composition replay. This is on-demand only;
the live trading loop, broker access, strategy settings and order flow remain
untouched. The preflight reads the already captured session and validates
writer integrity, process epochs, four-stream selected option/futures/spot
coverage, websocket provenance, two-sided bid/ask and five-level depth.

A missing, partial or corrupt archive produces a structured
`replay_capture_fidelity_preflight_unavailable` diagnostic, **not**
fabricated quote or fill data. The runtime replay still applies its original
strict checks and its own success/failure outcome. Even a high-quality
preflight is not broker-executable P&L and does not imply all exchange
messages arrived.

The production status relay only confirms writer counters; it does not expose
the archived JSONL bytes. Therefore the current day's raw depth fidelity
cannot be declared validated based on HTTP status alone. The automatic
preflight gives the existing on-demand replay path a consistent place to
record that evidence whenever a completed session is processed.


## Paired adaptive trailing-stop replay — research only

The same existing offline worker can now compare the current canonical
trailing policy against the optional adaptive profit-lock controller, using
the exact same checksum-verified recorded session and paper fill engine.
No new production service, broker connection, password or paid data
subscription is required.

On the machine holding the real session and its integrity sidecars, run:

```sh
python scripts/run_runtime_replay.py \
  --session /private/replay_archive/2026-10-21.jsonl \
  --output /private/research/paired-2026-10-21 \
  --compare-adaptive \
  --stress-bps 10 25 50
```

This command is strictly **on demand**. Each scenario runs two isolated,
credential-sanitized no-network subprocesses of the production-composition
replay: `ADAPTIVE_PROFIT_LOCK_ENABLED=false` (the current policy) versus
`true` (the research candidate). Environment and entry/exit choices remain
under the same captured strategy, risk and session policy. Replaying the
session does **not** activate adaptive trailing in the production engine.

Reports:

- `capture-fidelity.json`: the existing strict archive integrity and
  selected CE/PE/spot/futures quote-fidelity preflight;
- `baseline-10bps/report.json` and `adaptive-10bps/report.json` (and
  corresponding stress settings): modeled full-engine orders, costs and
  equity; and
- `paired-comparison.json`: matched-entry, matched-archive comparison
  with explicit blocker codes.

A numeric adaptive-vs-baseline delta is presented only if both runs use
identical archive SHA, event count, captured revision, **identical completed
BUY entries** (symbol, quantity, price and submission timestamp), have
complete observed selected quote context, and finish broker-simulation
flat. Different entries or open/partially filled positions are **UNRESOLVED**,
not silently dropped. A missing/incomplete replay tape blocks the comparison;
OHLC, MFE or a live-server status screenshot never substitutes for depth.

The actual paper broker model supplies order fees and adverse slippage. The
summary reports matched-entry net P&L, fees, modeled drawdown and sensitivity
to per-side slippage. This is still **not** validated exchange SL-LIMIT
execution, modification latency, queue-position feasibility, or proof of
broker-fill parity. It cannot be promoted automatically, even if all its
scenarios show a positive modeled delta. Require chronological non-overlapping
holdout days, validated costs, stop-limit gaps/partial-fill stress, and
independent live-market observation before considering live activation.
