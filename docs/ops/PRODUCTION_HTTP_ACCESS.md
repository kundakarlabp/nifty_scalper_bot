# Production HTTP access — canonical diagnostic route

**Last independently tested:** 2026-10-25 (IST). **Host:** AWS Lightsail `niftyscalper2`, Mumbai `ap-south-1a`, static public IPv4 `65.0.255.84`, private IPv4 `172.26.15.134`. **Do not route via AppDeploy** or assume the GitHub connector can administer AWS firewall rules.

## Canonical architecture

- **GitHub `main`** is the code/configuration source of truth. CI-validated changes are installed by the **single** guarded `deploy/lightsail_release.sh` on Lightsail; do not bypass broker-flat, terminal-order, and entry-hold gates.
- **AWS Lightsail** owns the runtime/systemd units and private credentials at `/home/ubuntu/.config/niftybot/niftybot.env`.
- **Supabase** project `dehdptgkqbrkyzyodicd`, table `public.nifty_runtime_heartbeat`, is a **read-only** off-host observer of the local engine, release, admin, and Streamlit services. This heartbeat is sent about every 120 seconds; it does **not** establish public internet reachability.
- **Do not use SSH or the retired AppDeploy relay to read live logs**. For HTTP inspection, first try direct public requests using a real external web client that can reach arbitrary IP ports. Some ChatGPT runtimes and Supabase's `extensions.http_get()` time out even when the endpoint is publicly reachable.

## Direct endpoints and expected semantics

| URL | Purpose | Off-hours `AUTO` result |
|---|---|---|
| `http://65.0.255.84:8080/livez` | Public engine process liveness | HTTP 200, `status=alive`, `quiet=true`, `bot_loaded=false` |
| `http://65.0.255.84:8080/releasez` | Public deployed GitHub SHA and release provenance | HTTP 200; compare `effective_sha` with GitHub main |
| `http://65.0.255.84:8501/_stcore/health` | Public Streamlit process liveness | HTTP 200, `ok` |
| `http://65.0.255.84:8080/health/trading` | Operational trading readiness | May be 404 while off-hours `AUTO` deliberately unloads the engine |
| `http://65.0.255.84:8080/admin` and `https://65.0.255.84/admin` | Password-free sanitized process-status home | Public process fields only; no broker tokens, positions, trade controls or privileged logs |
| `https://65.0.255.84/admin/manage` | Protected interactive trading operator dashboard | Trusted HTTPS and existing admin authentication required for credentials, LIVE switches, restarts and risk controls |
| `http://127.0.0.1:8081/healthz` | Host-local independent admin process liveness | HTTP 200, `status=ok`; never require public TCP 8081 |

On 2026-10-25, external HTTP from an independent browser/scrape client returned 200 on `:8080/livez`, `:8080/releasez`, `:8501/_stcore/health` **and** `:8081/healthz`. `/releasez` gave `runtime_sha=c86e6f627cb2`; Supabase confirmed full `c86e6f627cb24d97ed63dd9a6dc8f75fdf7ada21`, `release_state=current`, local HTTP 200, locally responsive admin and Streamlit. From the Supabase database-side HTTP extension, ports 80/443/8080/8501 timed out, even though other public sites succeeded: **this probe is not conclusive evidence that AWS is down or its port blocked**. An alternative external HTTP client conclusively disproved the 8080/8501 outage theory.

## AWS networking evidence and remaining security work

The owner's 2026-10-25 AWS console screenshots show `niftyscalper2` **running**, static IPv4 `65.0.255.84`, and inbound Lightsail IPv4 rules allowing TCP `80`, `8080–8081`, `22`, `8501`, `443` from any IPv4. `8080 -> 8081` on that screen is a **port range**, not a port-forwarding rule.

The independent HTTPS client could not negotiate trusted TLS on `:443` under the historical Caddy config (`:443 { tls internal ... }`), and `:8081` was public even though the admin server historically relied on firewall isolation rather than credentials. **Never treat that historical state as secure.** The desired boundary: leave harmless public liveness/diagnostic access on 8080; require encrypted authenticated access for `/admin/manage`, mutations and `/power`; the public `/admin` status home must contain only sanitized process data; bind the independent admin process to `127.0.0.1:8081`; use an ACME publicly trusted short-lived IP certificate on `443`, not Caddy's private `tls internal`. Let's Encrypt supports short-lived IP-address certificates since January 2026. Caddy **2.10.1+** supports IP identifiers; an explicit ACME `shortlived` profile is required. Verify public HTTPS and certificate validity **after** rollout, never assume issuance from a syntactically valid Caddyfile.

## Repeatable troubleshooting order

1. Verify GitHub `main` and the most recent successful **push-triggered** CI run for its exact SHA.
2. Read fresh Supabase heartbeat (`received_at`, `build_sha`, `http_status`, `trading_status`, `release_state`, `admin_http_responsive`, `streamlit_http_responsive`); check heartbeat age. `quiet`/`auto_quiet` with HTTP 200 is normal off-hours.
3. Probe external `/livez` and `/releasez` at `:8080`, and `/_stcore/health` at `:8501` using an independently network-enabled HTTP client (fresh requests, no cache). If one probe times out, try a second network vantage before alleging outage.
4. Check HTTPS `https://65.0.255.84/admin` (full controls after owner Zerodha sign-in) and legacy HTTP `:8080/admin` (HTTPS redirect, no Basic prompt) with certificate verification **enabled**. A TLS error is different from a TCP timeout; never enter credentials through insecure HTTP or `-k`.
5. If public TCP remains inaccessible from multiple independent networks, compare AWS **Networking > IPv4 firewall** with desired listeners, then check host `ufw`/`nftables`/actual systemd listener state using an authorized AWS management method. GitHub and Supabase cannot inspect the live AWS firewall or socket table.
6. Mark deployment complete **only** after Supabase SHA equals merged main and `release_state=current`. CI and HTTP liveness do not prove LIVE entry readiness. Never restart a LIVE-exposed engine merely to fix dashboard ingress.

This file is the SSOT for subsequent ChatGPT discussions of HTTP reachability; keep test claims dated and distinguish **verified**, **proposed**, and **not verified**.

## CI gate reliability and no-cost recovery

The 2026-10-25 incident also uncovered `release_state=ci_unavailable` despite
a completed, successful GitHub main-push CI run for the pending SHA. The
historical `verify_release_ci.py` used the unauthenticated GitHub Actions REST
API on a frequent release timer. An HTTP 403/429, rate limit, or host egress
failure yields `ci_unavailable`; **do not override the gate, restart trading,
or interpret this as failed tests**.

From the next merged revision onward, the full **push** CI workflow publishes
`refs/tags/nifty-ci-passed/<full-40-character-sha>` via the existing free
`GITHUB_TOKEN` only after `final-validation` succeeds. The host verifier
reads that exact non-forced ref using `git ls-remote` (the same transport used
by `git fetch origin main`), and requires the ref's target OID to match the
complete main SHA; the original GitHub API check remains as fallback. Missing
or inconsistent evidence fails closed. No Supabase write credentials, extra
service, GitHub PAT, or paid API is required.

The installed **old** release verifier cannot recognize newly added refs
before it successfully updates itself. If the old host remains
`ci_unavailable`, wait for its existing rate limit/network access to recover
or use an authorized AWS control-plane repair. **Never relax the old CI gate
through an unauthenticated external HTTP endpoint.** The guarded release
script preserves broker-flat and terminal-order confirmation. After the new
runner is installed, its regular timer can reconcile the Caddy/admin unit
from the GitHub checkout without restarting the engine; a failed repair is
recorded as operator-ingress degradation, not silently marked secure.

The CI proof tag shares GitHub repository write authority; protect that ref
namespace through GitHub rulesets if available, and restrict repository write
access. GitHub `main` remains the only release source.


## AUTO wake-up, Monday market session

`BOT_OPERATING_MODE=AUTO` loads only the lightweight control app outside
weekdays **08:55–15:40 IST**; the existing 30-second watchdog queues a
broker-safe validated service restart when crossing the operating boundary.
The first trading-entry opportunity is later than this warm-up window.
The watchdog must **retry a rejected queue attempt** on the next tick rather
than quitting and leaving an unloaded engine for the trading day. An
*accepted queue* still only means the release runner accepted the request;
it is not proof the broker session, feed, reconciliation, or protective
orders are ready. A fresh `/health/trading` readiness decision after
08:55 IST must confirm those requirements before any live entry.

On a normal off-hours Sunday, `/livez` returning
`{"quiet":true,"bot_loaded":false}` and Supabase
`trading_status=quiet` are expected. Do not activate the trading
engine manually on Sunday merely to prove Monday's broker authorization;
Zerodha access tokens are session-bound. Monday, 2026-10-26 is scheduled
as an NSE derivatives session. Do not infer live order permissions from
HTTP liveness or the absence of an exchange holiday.

## Single admin URL with owner-verified controls (26 October 2026)

One URL is canonical: `https://65.0.255.84/admin`. The former
`http://65.0.255.84:8080/admin` bookmark redirects to the same HTTPS
address; HTTP does not render broker credentials or permit mutations.
There is no separate bot username or Basic admin password. The same existing
full dashboard (`/admin`, also accessible as `/admin/manage`) includes
daily Kite login/request-token submission, access-token rotation, settings,
operating mode, logs, risk operations, and validated GitHub update/restart.

On the first access each day, the browser is directed to
`/admin/kite/start`: authenticate on Zerodha's official website, paste
the request token or redirect URL into that one form, and the server exchanges
the token and verifies the account against the **preconfigured**
`ADMIN_ZERODHA_USER_ID`. After that, the protected browser session opens the
full admin page directly without another username/password. Only the
existing authorized Kite account can open real-money operations. The
`ADMIN_ZERODHA_USER_ID` must already be securely set in the private host
environment; no anonymous enrollment or fallback is permitted. There is
no paid authentication service. The Zerodha broker API subscription, if
applicable, is separate from admin authentication.

A daily request token is one-time-use and expires; the application stores
verified broker access-token aliases in the existing protected host file
and **queues** the existing validated restart. A queued restart does
not prove broker readiness. Both the original daily token form inside the
full dashboard and the owner-login form continue to work.

The old HTTP Basic scheme, popup challenge and `ADMIN_PASSWORD` fallback
are not accepted by public admin routes. Cleartext access to privileged
API routes stays blocked. Do not remove the broker-flat, order-terminal,
entry-hold or risk controls: these are trading safety checks, not admin
logins, and there is no clock-based ban on intraday safe releases.

Verification after release:
- HTTP `:8080/admin` returns a 307 redirect to HTTPS; no Basic challenge.
- HTTPS `/admin` redirects an unknown browser to Zerodha login (303).
- HTTPS `/admin` opens the **full** editable dashboard for the signed owner.
- HTTP `/admin/manage` and unauthenticated mutations remain forbidden.
- Check actual AWS build SHA against merged GitHub main and complete release
  state before claiming the deployment is active.
