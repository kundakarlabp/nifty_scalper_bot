# NIFTY production — one-path operator runbook

**Single source of truth:** GitHub `main` contains code; AWS Lightsail is
the sole LIVE trading runtime; Supabase is the private read-only remote
telemetry store. **AppDeploy is not in the dependency chain.**
On 23 October 2026, the five obsolete AppDeploy NIFTY network probes/relays
were deleted and confirmed deleted. Do not recreate or redeploy them, including
during investigation of direct HTTP access failures. No bot order, release or
daily broker-token operation may require AppDeploy.

## Daily operation

- Open the existing Lightsail admin page (independent `niftybot-admin` on
  port 8081 behind the approved private/IP-restricted access path).
- If Zerodha is already authenticated and `/health/trading` shows no auth
  blocker, **do nothing**. Repeated token exchanges and restarts add risk.
- Otherwise use the official Zerodha login link once. The admin verifies,
  atomically saves the broker access-token aliases outside Git, and requests
  a **validated safe restart**. `queued` is not `completed`.
- Open positions, pending orders, unavailable broker verification, an active
  order-submit call, or an unacknowledged entry hold block restarting the
  LIVE trading engine. Protective exits are never blocked.
- Never paste the request-token URL, access tokens, or TOTP into chat,
  screenshots, GitHub Actions, or Supabase telemetry.

## Archived startup and recovery evidence (Phase 7)

The existing five-minute Lightsail journald archiver attaches a compact,
non-secret `startup_audit` to the already-present
`public.nifty_daily_market_logs.summary` JSONB column. Its existing Supabase
Edge receiver accepts only allowlisted counters, HH:MM:SS clocks and a fixed
`protective_stop_verification` caveat. Older clients without the audit
continue to archive logs without clearing existing summaries.

This is **session retrospective evidence**, not broker reconciliation authority.
Counts of `STARTUP_PHASE phase=create_context`,
`STARTUP_WS_START_DONE`, `RECONCILE_COMPLETE`, and
`STARTUP_MODE_TRANSITION ... to=LIVE_READY` help investigate repeated
intraday restarts. The markers **cannot prove exchange-side stop protection**
or explain the root cause of a restart. Missing markers are missing evidence,
not success. Partial intraday archives may change; use `finalized` when
comparing complete sessions.

Read-only verification:

```sql
select trading_date, finalized, line_count,
       summary -> 'startup_audit' as startup_audit
from public.nifty_daily_market_logs
order by trading_date desc
limit 5;
```

Investigate repeated intraday startups against broker reconciliation evidence.
Do not restart services or change trade/risk settings based on this summary.

## Releases — ONE production deploy authority

GitHub CI pass on the exact merged `main` SHA → existing
`niftybot-autodeploy.timer` on Lightsail (~30 seconds after completed
prior validation) → `deploy/lightsail_release.sh --auto` → broker-flat
handoff → install + restart + readiness verification (or rollback).

**Do not add GitHub-hosted trading services, GitHub deployment Actions,
AppDeploy deploy triggers, extra daily restarts, webhook listeners, or
alternate restart logic.** The admin's GitHub Update button only requests
an immediate check of the SAME Lightsail deployer; the timer is already
automatic. CI green is necessary, not sufficient, to prove a live release.

Release source of truth on Lightsail:
`data/auto_update_status.json` (state and updated_at), supplemented by
`/releasez` and the running engine's build SHA. A valid release is
**the running process matching the intended main SHA and passing health**.
Do not infer it from a button, commit, or queued deploy marker.

## LIVE risk ceiling and operator overrides

GitHub's `config/defaults.py` retains the existing **5% per-trade** and
**5% daily-loss** risk ceilings; this rollout does not increase or recommend
either limit. The canonical `config/env_utils.py` normalizer now respects
explicit *lower* operator limits instead of silently overwriting them with 5%.
Conflicting aliases resolve to the strictest finite positive value, while
attempted increases above GitHub's ceiling are capped. Invalid, zero, negative,
NaN or infinite LIVE limits fail closed before broker-live authorization.

The effective per-trade and daily values are logged as
`LIVE_RISK_ENVELOPE_CONFIG`, with no account balance, token or other secrets.
This is a safety correction, not a strategy or trading-exposure optimization.
Any later decision to revise the default 5% profile needs a separate risk
evaluation; do not silently alter active positions or protective orders.

## One protected production environment source

The runtime settings loader reads only `BOT_ENV_FILE` when that explicit
file is provided. In the Lightsail production profile, an absent or invalid
`BOT_ENV_FILE` fails closed to SHADOW; it never tries a project, working
directory or fallback `.env` that could accidentally authorize LIVE.

Values already supplied by the systemd `EnvironmentFile` retain precedence
(`override=False`). The protected file contains broker credentials and
explicit operator mode/override settings; GitHub continues to own strategy
defaults. Local development without a production profile still supports its
historic `.env` discovery. This is not a credential migration and never
deletes or rewrites existing host settings.


## Remote operator-service liveness (Phase 6)

The existing `RuntimeHeartbeat` reporter checks two **fixed loopback
addresses** during its off-trading-hot-path 120-second Supabase update:
`127.0.0.1:8081/healthz` for the independent admin process and
`127.0.0.1:8501/_stcore/health` for Streamlit. Both require the expected
HTTP 200 response body and do not follow redirects. No new daemon, token,
gateway, public firewall port, AppDeploy relay, or control RPC is involved.

The same private `nifty-runtime-heartbeat-ingest` receiver accepts only
booleans or null for `admin_http_responsive` and
`streamlit_http_responsive`, and saves them alongside the existing engine
SHA, release status and market-session state. Missing values from old
releases remain **null / unknown**, not false. The two nullable columns
are added by the versioned Supabase migration in this repository. The
migration and backward-compatible Edge receiver must be present before
deploying the new sender on Lightsail; otherwise monitoring should fail
closed without affecting order execution.

Read-only verification:

```sql
select received_at, build_sha, release_state, trading_status,
       admin_http_responsive, streamlit_http_responsive
from public.nifty_runtime_heartbeat
order by received_at desc
limit 1;
```

`false` means the specific HTTP service did not respond correctly **from
Lightsail localhost** during the probe. It is not evidence of a public
network outage. `true` proves only that service's local HTTP liveness,
not broker authentication, safe LIVE entry authorization or externally
accessible UI. `null` means missing/invalid/not-yet-upgraded evidence.
The `received_at` timestamp must be fresh before interpreting any status.

## GitHub-owned telemetry defaults

Supabase monitoring policy and nonsecret Edge ingestion URLs now live in
`src/nifty_scalper_bot/config/defaults.py` and
`config/env_utils.py:supabase_telemetry_enabled`. The authorized Lightsail
runtime enables heartbeat, journald archive and trade-ledger replication
automatically when the respective setting is absent. Development and other
hosts default to disabled; explicit `false` overrides remain available.

Neither `deploy/lightsail_setup.sh` nor `deploy/lightsail_release.sh` now
duplicates those endpoint URLs, timing values, or writes telemetry settings
into the protected environment file. Existing AWS overrides are deliberately
preserved; deleting unknown historical entries requires a separate effective
config comparison and must not silently change trading behavior. Supabase
remains a monitor, not a release, order or risk controller.

## Monitoring — ONE private remote source

The pre-existing Supabase project `dehdptgkqbrkyzyodicd` contains:
- `public.nifty_runtime_heartbeat`: newest engine status, SHA, and
  allowlisted release state + timestamp; refreshed about every 120 seconds
- `public.nifty_trade_events`: replicated broker-trade events
- `public.nifty_daily_market_logs`: incremental and finalized day logs

A heartbeat older than **5 minutes** is **STALE/UNKNOWN**, not necessarily a
stopped engine. A fresh `trading_status=quiet` during AUTO off-hours is
a planned idle, not an outage. A fresh `armed` heartbeat does not replace
the authenticated broker checks for placing any trade. The release fields
are read from the host's canonical release status; never accept status text
as an order or action instruction.

Read-only monitoring query for the authorized Supabase SQL connector:

```sql
SELECT now() AS checked_at,
       observed_at, received_at, build_sha, engine_healthy,
       trading_status, blockers, release_state, release_updated_at
FROM public.nifty_runtime_heartbeat
WHERE source = 'lightsail';
```

The dashboard `/admin/api/overview` and `/admin/status.json` use live
local HTTP health and the same release-state file; **not historical log
phrases**. Heavy raw log downloads remain on demand.

## When something actually fails

| Symptom | Authority to check | Correct response |
| --- | --- | --- |
| AppDeploy 402 | Supabase heartbeat | Ignore relay billing for bot health; never restart engine just for 402 |
| Bot not trading | Admin broker/readiness + local logs | Identify actual reason; don't automatically change access token |
| GitHub merged but old SHA | Canonical release state, CI, 30-second timer | Distinguish deferred order exposure, failed CI, or inactive timer |
| Dashboard unreachable | Admin :8081 `/healthz` | Recover independent admin service, not the trading engine |
| Stale heartbeat | Supabase receiver, local engine and network | Report UNKNOWN until local health and transport verified |
| Quiet out of hours | `/livez` | Expected; do not treat as failure |

Production engine port 8080, admin 8081, optional read-only Streamlit 8501.
Do not expose write-capable admin interfaces openly on the internet;
the legacy admin relies on an external network allowlist rather than
built-in user authentication. Preserve the existing allowlist.

## Minimal AWS process inventory

Keep the **three existing, distinct processes** for fault containment, not a
larger constellation of duplicate services:

| Purpose | Service | Critical? |
| --- | --- | --- |
| LIVE trading and protective exits | `niftybot` :8080 | Yes |
| Daily broker login, manual actions and independent recovery | `niftybot-admin` :8081 | Yes |
| Read-only reports / operator console | `niftybot-streamlit` :8501 | Optional for orders; preserve current user features |
| Verified code-release poll | `niftybot-autodeploy.timer` | Yes |

Do **not** restart the trading engine merely because the optional
Streamlit page or a monitoring provider fails. Never put an order or secret
in the read-only console. Optional Docker/Prometheus/Grafana scripts are
research infrastructure, not an additional deployed LIVE bot.

## Removed third-party relay inventory

These obsolete AppDeploy apps were permanently deleted on 23 October 2026,
with `status=deleted` independently checked:
`nifty-endpoint-probe-qspiq0`,
`nifty-deploy-recovery-relay-u7c57n`,
`nifty-cloud-probe-capture-w3mrp1`,
`nifty-live-diagnostic-relay-09rb32` and
`nifty-live-readonly-relay-kb0r4e`.

The separate `nifty-historical-data-collector-pwtdwk` is **not** a LIVE
dependency but still stores Dhan history in AppDeploy storage. Preserve and
export/verify the archive before deleting that app. Three unrelated clinical
apps also remain and must not be silently destroyed by NIFTY maintenance.

For first-time Lightsail host provisioning, see `deploy/LIGHTSAIL_SETUP.md`.
For the separate research/backtesting environment see
`docs/deployment/README.md`. Do not confuse Docker research deployment with
production Lightsail deployment.
