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
| `https://65.0.255.84/admin` | Sensitive authenticated operator dashboard | Requires a valid public TLS certificate and admin authentication; **do not send credentials over HTTP** |
| `http://127.0.0.1:8081/healthz` | Host-local independent admin process liveness | HTTP 200, `status=ok`; never require public TCP 8081 |

On 2026-10-25, external HTTP from an independent browser/scrape client returned 200 on `:8080/livez`, `:8080/releasez`, `:8501/_stcore/health` **and** `:8081/healthz`. `/releasez` gave `runtime_sha=c86e6f627cb2`; Supabase confirmed full `c86e6f627cb24d97ed63dd9a6dc8f75fdf7ada21`, `release_state=current`, local HTTP 200, locally responsive admin and Streamlit. From the Supabase database-side HTTP extension, ports 80/443/8080/8501 timed out, even though other public sites succeeded: **this probe is not conclusive evidence that AWS is down or its port blocked**. An alternative external HTTP client conclusively disproved the 8080/8501 outage theory.

## AWS networking evidence and remaining security work

The owner's 2026-10-25 AWS console screenshots show `niftyscalper2` **running**, static IPv4 `65.0.255.84`, and inbound Lightsail IPv4 rules allowing TCP `80`, `8080–8081`, `22`, `8501`, `443` from any IPv4. `8080 -> 8081` on that screen is a **port range**, not a port-forwarding rule.

The independent HTTPS client could not negotiate trusted TLS on `:443` under the historical Caddy config (`:443 { tls internal ... }`), and `:8081` was public even though the admin server historically relied on firewall isolation rather than credentials. **Never treat that historical state as secure.** The desired boundary: leave harmless public liveness/diagnostic access on 8080; require encrypted authenticated access for `/admin` and `/power`; bind the independent admin process to `127.0.0.1:8081`; use an ACME publicly trusted short-lived IP certificate on `443`, not Caddy's private `tls internal`. Let's Encrypt supports short-lived IP-address certificates since January 2026. Caddy **2.10.1+** supports IP identifiers; an explicit ACME `shortlived` profile is required. Verify public HTTPS and certificate validity **after** rollout, never assume issuance from a syntactically valid Caddyfile.

## Repeatable troubleshooting order

1. Verify GitHub `main` and the most recent successful **push-triggered** CI run for its exact SHA.
2. Read fresh Supabase heartbeat (`received_at`, `build_sha`, `http_status`, `trading_status`, `release_state`, `admin_http_responsive`, `streamlit_http_responsive`); check heartbeat age. `quiet`/`auto_quiet` with HTTP 200 is normal off-hours.
3. Probe external `/livez` and `/releasez` at `:8080`, and `/_stcore/health` at `:8501` using an independently network-enabled HTTP client (fresh requests, no cache). If one probe times out, try a second network vantage before alleging outage.
4. Check HTTPS `https://65.0.255.84/admin` with certificate verification **enabled**. A TLS error is different from a TCP timeout; never enter credentials through insecure HTTP or `-k`.
5. If public TCP remains inaccessible from multiple independent networks, compare AWS **Networking > IPv4 firewall** with desired listeners, then check host `ufw`/`nftables`/actual systemd listener state using an authorized AWS management method. GitHub and Supabase cannot inspect the live AWS firewall or socket table.
6. Mark deployment complete **only** after Supabase SHA equals merged main and `release_state=current`. CI and HTTP liveness do not prove LIVE entry readiness. Never restart a LIVE-exposed engine merely to fix dashboard ingress.

This file is the SSOT for subsequent ChatGPT discussions of HTTP reachability; keep test claims dated and distinguish **verified**, **proposed**, and **not verified**.
