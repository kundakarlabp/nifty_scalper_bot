# Read-only live replay capture fidelity audit

`src/nifty_scalper_bot/backtesting/replay_fidelity.py` is a streaming,
read-only preflight of *already-recorded* production market data. It calls
`validate_replay_epochs` and `iter_session` before presenting quality counts.

For a **completed local archive day**, run:

```bash
python -m nifty_scalper_bot.backtesting.replay_fidelity \
  --session /path/to/existing/replay_archive/2026-10-21.jsonl \
  --output /path/to/private/audit-2026-10-21.json
```

The input must be accompanied by the replay recorder's existing
`<run_id>.integrity.json` sidecar file(s). Do not copy a partial live
recording and infer that a read failure means the bot has stopped; the writer
updates its counters asynchronously. The audit **does not** fetch a file from
the production host or via an unsupported HTTP route. It never calls a broker,
adds an exchange order, changes risk, or deploys production code.

## Evidence gate

- Capture sequence and final count integrity: gap/truncation or dropped/failed
  writes lead to a hard failure.
- Snapshot validity: initial balance, instrument master, selected basket,
  configuration and immutable process state are checked using the existing
  production-replay validator.
- Observe selected CE/PE, spot and futures, count WS versus non-WS delivery,
  symbol token provenance, two-sided uncrossed quotes, complete five-level
  order-book observations and intertick gaps.
- Analyze quote **observations** only. Five-level websocket depth is not
  complete order-by-order exchange depth; no negative conclusion about an
  illiquid contract can be drawn solely from absent ticks. A tick gap is
  not evidence that an exchange event was dropped.
- Any changing selected CE/PE/futures basket causes a conservative
  `observed_selected_context_available=false` until more granular
  basket-phase replay proof exists. It does not flatten basket phases into
  a retrospectively invented continuous contract.
- Never compute account P&L from LTP, OHLC, or raw best bid/ask alone. No
  historical fee calibration, order queue position, partial fill or
  exchange reject is inferred.

The report stores only aggregate quality counts, the archive SHA-256 and
session date. It **does not expose account credentials, individual quote
prices, raw historical order books or instrument catalog rows.**

### 21 October 2026 read-only live status check

The existing status relay exposed **158,575 recorded events**, **zero
reported dropped/failed writes**, **one pending writer event**, and
`runtime_replay_completed=false` during the observed snapshot. Those
counters establish that recording was active at that moment, **not** that
the resulting raw file has verified replay fidelity. The production code
revision reported `eee8cf5246034259a41e27d25571796d09ad2648`; recent
research-only merges were not independently verified as deployed.

A complete live-equivalent backtest still requires replaying the archived
delivery sequence with the real captured strategy/risk state, calibrated costs,
measured fill feasibility, an untouched holdout, and actual broker fills
for model comparison. No strategy promotion is authorized by this audit.
