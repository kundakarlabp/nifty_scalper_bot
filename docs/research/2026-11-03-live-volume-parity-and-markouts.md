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
- `REPLAY_CAPTURE_MAX_SESSION_MB` requests 1,024 MiB by default and
  is clamped to 128–2,048 MiB. A one-time recorder startup disk check retains
  a 2 GiB free-space reserve; otherwise the recorder fails back to the
  original 512 MiB ceiling and publishes `capacity_guard` plus a warning.
  The sidecar records the effective ceiling. Capacity exhaustion still
  invalidates replay; never silently sample away ticks.
- The live strategy manager emits `OPTION_COMPLETED_VOLUME_PARITY` once each
  UTC minute for **both** selected CE/PE. It compares completed minute
  timestamps and actual volume across the indicator and canonical data paths.
  Equal *zero* is `zero_volume_match_unverified`, **not** verified parity.
  Missing or different-minute pairs are counted separately. When one
  source has not yet finalized the newest bar, the observer compares the newest
  common completed minute and separately records both latest source timestamps
  and `latest_sources_aligned`; a historical match does not certify live
  source synchronization. Genuine same-minute differences remain mismatches.
- Independently of strategy acceptance, `RESEARCH_STRATEGY_REJECTION`
  records the first observed SMC, VWAPPro and ORBPro no-vote per UTC minute,
  exact contract and strategy. The sampled event now also stores finite,
  allowlisted decision-time option premium, VWAP, ATR, direction,
  bid/ask/spread metadata, selected contract and observed lot size when present.
  It never logs raw depth arrays, credentials or arbitrary private fields. This is **not** a structural setup ID; samples
  must not be called independent trading opportunities.
- The existing five-minute log archive adds `REPLAY_CAPTURE_STORAGE` status
  (file size, ceiling, queue backlog and integrity counters). No tape bytes,
  broker tokens or private account data are uploaded. A strict
  post-session audit now publishes exact missing selected CE/PE tick-update
  minute ranges and separate stricter two-sided WebSocket quote-update ranges.
  REST-only and LTP-only ticks do not establish executable quote coverage;
  zero writer drops does not establish exchange completeness.

A replay used in two separate research/fidelity passes must have the same
SHA-256 after both passes; differing hashes invalidate the modelled markouts
and block any implication of executable strategy expectancy.

## Post-close read-only report

After 15:33 IST (three-minute writer settle window), the existing log
archiver processes the original local replay with the strict integrity
sidecars. It then computes bid/ask observations at 60, 180, 300, 600 and 900 seconds
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

## Compact retention and cost-adjusted 10/15-minute evidence

The existing canonical replay tape remains the sole high-frequency tick/depth
store; the existing minute-bounded journald/Supabase observation stream remains
the only no-vote research journal. No additional SQLite tick database is added.

Off-market maintenance (REPLAY_ARCHIVE_COMPACTION_ENABLED=true by default)
checks prior-day or older sessions only outside 07:00–17:00 IST. Original
writer integrity and event sequence are checked before compression; the
decompressed SHA-256 and replay sequence must match before raw source removal.
The active current-day file is untouched. Invalid days are retained. Historical
runtime replay and post-market research accept validated .jsonl.gz files.

Post-close markouts now include 1, 3, 5, 10 and 15 minutes for rejected
minute samples and structurally identified setups. Separate modeled one-lot
costs use the actual observed ask to enter and bid to exit, canonical broker
charges, and an explicit additional adverse slippage assumption of 5 basis
points per side. If contemporaneous instrument lot size is absent, 65 units
is a research-only default assumption, NOT exchange-verified position sizing.

Modeled net P&L and MAE/MFE require a complete quote path with no interior
gap above ten seconds; sparse gross labels remain gross-only. No bid/ask
series is reconstructed from LTP, and the spread is not charged twice.
Broker fills, executable quote quantity, exchange stops and actual stop-outs
are not verified. A no-vote has no inferred protective stop. Results remain
observational and cannot justify modifying LIVE admission or risk thresholds.

The original host-local JSON research report and bounded
RESEARCH_MARKOUT_STRATUM logs are reused. The file remains inspectable for
on-demand analysis, while summarized strata flow through the existing
authorized Lightsail to Supabase archive. No new external credentials, new
service or public endpoint is required.
