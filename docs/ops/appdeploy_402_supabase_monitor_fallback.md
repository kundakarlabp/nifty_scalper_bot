# Historical incident: AppDeploy HTTP 402 (relay retired)

**Archived for diagnosis, not an operating procedure.** The five obsolete
NIFTY AppDeploy relays were deleted on 23 October 2026. Do not restore or
recreate them. For live operations, use the single-path
[Lightsail + Supabase runbook](nifty_single_path_runbook.md).
The retired relay should never be treated as the primary bot health check.

---


## Verified diagnosis on 22 October 2026

The AppDeploy account's backend gateway returned
`{"code":"APP_TEMPORARILY_UNAVAILABLE"}` with HTTP **402** for **four
unrelated apps** (including the NIFTY relay), while their frontend pages
loaded and deployment control-plane reported `ready`. The account was
Free with ~588 **monthly** credits still available, but daily/weekly limits
are independent. Published Free limits: 100/day, 400/week, 1000/month.
This is consistent with a daily or weekly quota restriction; the account API
does not identify which is exhausted. No code change or redeployment can
restore an account-wide AppDeploy credit allowance.

AppDeploy credit limits: https://widget.appdeploy.ai/pricing

**Do not restart the bot or redeploy AppDeploy merely because the gateway
returns 402.** Engine :8080 direct reachability from a sandbox may be
blocked independently. Do not use SSH for read-only status inspection.

## Existing independent source: archived logs

Production already sends full market-session journals through a dedicated
Supabase Edge ingest to private `public.nifty_daily_market_logs`.
At 23:24 IST on 22 October, the 22 October archive was independently
readable, finalized, with 21,830 lines and 4,925,838 bytes; the final
archive write was at 17:24 UTC. Logs remain accessible during AppDeploy 402.

Use the authorized Supabase connector's SQL execution for project
`dehdptgkqbrkyzyodicd`:

```sql
SELECT trading_date, captured_at, finalized, line_count, build_sha
FROM public.nifty_daily_market_logs
ORDER BY trading_date DESC LIMIT 3;
```

For bounded evidence, avoid sending entire multi-MB raw archives:

```sql
SELECT right(log_text, 10000) AS latest_market_logs
FROM public.nifty_daily_market_logs
WHERE trading_date = current_date;
```

Use explicitly intended IST trading date rather than `current_date` if
Postgres timezone is UTC. Archive freshness is not engine liveness.

## Independent lightweight heartbeat, every 120 seconds

The bot's local FastAPI lifespan starts an isolated background task that
reads **only** `http://127.0.0.1:$PORT/health/trading` (2s timeout) and
sends sanitized status to the Supabase Edge function
`nifty-runtime-heartbeat-ingest`; no broker credentials, balances, quotes,
positions, order IDs or full logs are sent. It is disabled by default in
development and enabled by Lightsail's existing release script.

The Edge function accepts POST from the fixed Lightsail IP, requires its
source marker and a fresh observation timestamp, and upserts to
`public.nifty_runtime_heartbeat`. RLS is enabled and table access revoked
for `anon` and `authenticated`. The receiver cannot affect trading.

```sql
SELECT source, observed_at, received_at, build_sha, engine_healthy,
       trading_status, blockers, failure_reason
FROM public.nifty_runtime_heartbeat WHERE source = 'lightsail';
```

Interpret freshness explicitly: `now() - received_at < interval '5 minutes'`
means a recent emitter; an old row is **STALE/UNKNOWN**, not conclusive proof
that the bot is down. `engine_healthy=true` and
`blockers=["market_closed"]` means the engine answered but entries are
blocked appropriately by time. Missing row means never verified.

## Validation after main merges

1. GitHub PR CI must pass before merging.
2. Confirm production build SHA independently when heartbeat arrives; do not
   infer deploy from GitHub merge alone.
3. Verify `received_at` advances at least twice, and health snapshot is
   fresh. The Edge source-IP validation must pass in production.
4. Check existing full archive ingest remains successful.
5. Confirm no modification to trader risk, order submission, API tokens,
   AppDeploy app versions, or paid services.

**Scope:** this is a durable Supabase monitoring fallback. AppDeploy's own
backend HTTP service remains restricted while its account quotas are
unavailable and is expected to resume when quota permits. No workaround
should masquerade as restoring the AppDeploy gateway itself.
