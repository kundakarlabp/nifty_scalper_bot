# Live option-volume parity and rejected-setup markouts

**Owner:** existing MarketDataManager replay tape + StrategyManager evidence.
**Transport:** existing Lightsail journald → Supabase daily log archive. No new
WebSocket, database schema, public endpoint, automation service or broker call.

## Recording contract

- `REPLAY_CAPTURE_ENABLED=true` (existing default). Record every delivered
  normalized WebSocket quote with original `available_at`, bid, ask, depth,
  source, symbol and snapshot context in `DATA_DIR/replay_archive/YYYY-MM-DD.jsonl`.
- `REPLAY_CAPTURE_QUEUE_CAPACITY` defaults to 16,384 and is clamped to
  1,024–65,536. Queue drops invalidate integrity. Never block trading to
  preserve research.
- `REPLAY_CAPTURE_MAX_SESSION_MB` defaults to 512 and is clamped to
  128–2,048 MiB. Keep the default until host disk headroom is verified.
  Capacity exhaustion invalidates replay; never silently sample away ticks.
- The live strategy manager emits `OPTION_COMPLETED_VOLUME_PARITY` once each
  UTC minute for **both** selected CE/PE. It compares completed minute
  timestamps and actual volume across the indicator and canonical data paths.
  Equal *zero* is `zero_volume_match_unverified`, **not** verified parity.
  Missing or different-minute pairs are counted separately.
- Independently of strategy acceptance, `RESEARCH_STRATEGY_REJECTION`
  records the first observed SMC, VWAPPro and ORBPro no-vote per UTC minute,
  exact contract and strategy. This is **not** a structural setup ID; samples
  must not be called independent trading opportunities.
- The existing five-minute log archive adds `REPLAY_CAPTURE_STORAGE` status
  (file size, ceiling, queue backlog and integrity counters). No tape bytes,
  broker tokens or private account data are uploaded.

## Post-close read-only report

After 15:33 IST (three-minute writer settle window), the existing log
archiver processes the original local replay with the strict integrity
sidecars. It then computes bid/ask observations at 60, 180 and 300 seconds
for each recorded no-vote minute and for independently identifiable
`candidate.approved/candidate.blocked` setups from the canonical
`DATA_DIR/trades.db` journal.

Each markout requires the exact unchanged option contract, point-in-time
WebSocket ask no older than three seconds, at least three subsequent usable
bid quotes, and a terminal bid no older than five seconds at the horizon.
`ws_full` **and** `ws` sources are accepted; REST, stale, absent depth and
rotated contracts are never substituted. Missing labels remain explicit.

Full report: `DATA_DIR/research/markouts/YYYY-MM-DD.json`. It includes the
replay SHA-256, individual sampled outcomes, independent structural setup
outcomes and coverage, parity counts and capture status. Atomic replacement
prevents publishing half a report.

The summary events `RESEARCH_SESSION_EVIDENCE` and
`RESEARCH_MARKOUT_STRATUM` are appended to the *existing* Supabase daily
`nifty_daily_market_logs.log_text`; no Supabase schema or Edge Function
changes are needed. An unavailable tape or failed integrity check produces
a blocked report, **not** invented forward returns.

Sampled-minute strata never count as structurally independent; even ten
labelled observations permit description only, not causal gate changes.
Broker fill probability, realized fees, slippage and stop policy are not
proved by ask-to-bid quote marks. Entry and risk settings remain unchanged.

## Read-only checks

From Supabase SQL, first inspect current deployment:

```sql
select observed_at, build_sha, engine_healthy, release_state
from public.nifty_runtime_heartbeat;
```

To count sampled records in the latest trading session:

```sql
select trading_date, captured_at,
       regexp_count(log_text, 'OPTION_COMPLETED_VOLUME_PARITY') as parity_rows,
       regexp_count(log_text, 'RESEARCH_STRATEGY_REJECTION') as rejection_rows,
       regexp_count(log_text, 'REPLAY_CAPTURE_STORAGE') as capture_checks,
       regexp_count(log_text, 'RESEARCH_MARKOUT_STRATUM') as markout_strata,
       regexp_count(log_text, 'RESEARCH_SESSION_EVIDENCE') as report_events
from public.nifty_daily_market_logs
order by trading_date desc limit 5;
```

Check actual archived `OPTION_COMPLETED_VOLUME_PARITY` JSON lines for
matching contract and UTC minute on each side, positive counts, and missing
or mismatch statuses. Ensure `REPLAY_CAPTURE_STORAGE` reports no drops or
failures and sustainable disk headroom. A green CI test alone is **not**
live parity certification.

For failed quote labels inspect the local report's `label_status` values:
`no_tradable_exact_contract_quotes`, `no_point_in_time_entry_ask`,
`insufficient_forward_quotes`, `stale_terminal_bid`, or
`contract_rotation_or_missing_identity`.

Do not relax VWAP distance, SMC sweep/confirmation, ORB timing, or risk
filters solely because sampled no-votes have positive gross markouts.
A predeclared multi-session holdout with adequate independent structural
setups, real broker costs, depth, stop exposure and missing-data sensitivity
is required before promotion.
