# NIFTY production diagnostics: relay outage and independent fallback

## Scope and classification

Production engine: `65.0.255.84:8080`; admin: `:8501`.
AppDeploy diagnostic relay (existing v15; preserve deployment):
`https://api-v2.appdeploy.ai/app/nifty-endpoint-probe-qspiq0/api/nifty/`.
On 2026-11-19, `/livez`, `/health`, `/status`, and `/logs` returned HTTP 404
with body `{"error":"Backend not found"}` from AppDeploy. The platform host
also returned its own 404 for an unrelated `/health` path. A relay 404 is a
routing/backend-binding failure, **not proof that the trading engine is down**.
Do not automatically redeploy AppDeploy v15 merely to read production logs.
Never SSH to recover diagnostics. Check authorization, backend binding, release
revision, active runtime and app route with the AppDeploy owner before changing it.

## Independent read-only evidence source (available today)

The live Lightsail reporter writes heartbeat and periodic, partial market-day
journald archives into Supabase project `dehdptgkqbrkyzyodicd`.
Use an authenticated Supabase SQL connection; do not expose these logs publicly.

### Heartbeat: engine/streamlit health and build identity

```sql
select observed_at, received_at, source, build_sha,
       engine_healthy, http_status, trading_status, blockers,
       release_state, admin_http_responsive, streamlit_http_responsive
from public.nifty_runtime_heartbeat
order by observed_at desc
limit 1;
```

### Archive freshness and candidate suppression

```sql
select trading_date, captured_at, last_log_at, build_sha, finalized,
       line_count, (summary->'startup_audit'->>'observed_startup_count')::integer
           as observed_startups
from public.nifty_daily_market_logs
order by trading_date desc limit 3;
```

### Bounded latest 500 runner lines (archive, not guaranteed to be real-time)

```sql
with archived as (
    select log_text, captured_at
    from public.nifty_daily_market_logs
    order by trading_date desc limit 1
),
lines as (
    select captured_at, line_no, log_line
    from archived
    cross join lateral regexp_split_to_table(log_text, E'\\n')
        with ordinality as l(log_line, line_no)
)
select captured_at, line_no, left(log_line, 1500) as log_line
from lines where log_line like '%RUNNER%'
order by line_no desc limit 500;
```

The archive is a delayed observer; `captured_at` and `last_log_at` must be shown
when claiming freshness. A healthy heartbeat and an old archive do not prove
current live evaluation. Do not infer order-book integrity or broker positions
from an HTTP health code.

## WebSocket 1006 during application shutdown

2026-11-19 10:10:40 IST: `Stopping niftybot.service`.
10:10:41-42: application/runner orderly shutdown.
10:10:43: KiteTicker transport emitted 1006 close-handshake timeout; no independent
evidence of a spontaneous mid-trade market-data outage.
10:10:50: replacement build launched; startup completed.
Classify shutdown callbacks separately from unexpected active-session 1006.
Unexpected active disconnects must still fail closed and reconnect.

## Candle-history diagnostic contract

`RUNNER_HISTORY_CONSISTENCY` must report **both** count-based readiness
and an actual bounded comparison of canonical MDM, runner projection and
indicator projection, aligned by completed UTC minute. Compare open/high/low/
close/volume with a fixed numeric tolerance, not only the number of candles.
If a projection differs, reseed from canonical and inspect again. Never
overwrite canonical CandleEngine with runner-derived values, synthesize a
missing candle, or treat a repaired divergence as a profitable entry.

## Risk calculation versus operator rupee ceiling

As of 2026-11-19, available funds were INR 15,699.00 and source-configured
LIVE per-trade/daily policies were 5%; INR 784.95 is exact arithmetic.
The previously stated INR 773.00 ceiling is **not** equivalent. The existing
`RiskManager._configured_daily_loss_cap` does enforce the smaller of percent
and positive absolute cap, via the `RISK__MAX_DAY_LOSS`/
`RISK_MAX_DAY_LOSS` operator env setting.

If the INR 773.00 absolute ceiling is authoritative, configure
`RISK__MAX_DAY_LOSS=773.0` in the production operator environment after
reconciling live broker positions/pending orders and protective exits.
**Changing a GitHub example file does not modify the running environment**.
Confirm the running risk snapshot and `max_day_loss` before stating that the
ceiling is enforced. On an already-traded day, the risk circuit intentionally
freezes its original daily cap to avoid rebasing against reduced free cash;
a tightened policy requires explicit safe cutover validation.
