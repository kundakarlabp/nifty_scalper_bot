# Daily token and release pipeline — 23 October 2026

## Expected single operator workflow

1. Open the independent admin at port 8081 (or the approved reverse-proxy URL).
2. Click official Zerodha login; enter credentials only at Zerodha.
3. Admin exchanges request_token, verifies the resulting access_token with Kite
   /user/profile, and atomically saves aliases in the external env file.
4. A short-lived SHA-256 proof of the newly verified token is stored outside Git
   before queueing the existing canonical release/restart service.
5. The release runner holds new BUY submissions, then twice checks the actual
   broker positions and all nonterminal orders using the NEW token.
6. Only if flat/terminal and the running engine acknowledges the release entry
   hold does it restart. The engine reads new access token on startup.
7. The admin reports QUEUED separately from DEPLOYED / engine restarted.
   Broker authentication must be checked on the NEW process before calling it
   applied. A queued request is never success.

A normal dashboard Restart/Update button has NO verified rotation proof and
cannot bypass the current engine's broker authentication check. Stale/reused
verification proofs are refused. A valid token alone never authorizes restarting
with open positions or active orders.

## Known blockers distinguished

- A token submitted on the dashboard is not automatically loaded by the already
  running process: restart is required, and is separately safety checked.
- Previous runner demanded that the old process prove a still-valid session.
  On daily rotation, the old token may have expired although the new token was
  verified and exchange positions/orders are flat. Only the token-bound proof
  now permits ignoring that obsolete *old-process* auth bit.
- The release timer must be active. GitHub merge alone cannot turn on a disabled
  Lightsail systemd service.
- For a code update, exact *main-push* CI must pass before install. Token refresh
  does not grant permission to ignore required CI when a newer code revision is
  pending.
- AUTO mode intentionally loads only the small control app outside 08:55–15:40
  IST; it has no /health/trading. Lightweight runtime heartbeat detects /livez
  with quiet=true and reports status=quiet, not a dead engine.
- AppDeploy gateway HTTP 402 affected unrelated apps; it is NOT a bot release
  failure. Keep read-only logs and heartbeat in private Supabase instead. Do
  not redeploy AppDeploy solely to read logs.

## Inspectable read-only checkpoints

On the Lightsail host, with approved interactive access, the canonical sources
are (do not send environment-file contents or access tokens):

```bash
systemctl is-active niftybot niftybot-admin niftybot-autodeploy.timer
systemctl list-timers --all | grep niftybot-autodeploy
curl --max-time 3 -fsS http://127.0.0.1:8081/healthz
curl --max-time 3 -fsS http://127.0.0.1:8080/livez
cat /home/ubuntu/nifty_scalper_bot/data/auto_update_status.json
```

Only "deployed" with the correct SHA and a healthy restarted process proves
deployment; requesting a restart/update never does. The private Supabase
`nifty_runtime_heartbeat` is useful when fresh, and must be labelled STALE
when older than 5 minutes. An empty table is UNKNOWN, not a bot outage.
The archived `nifty_daily_market_logs` proves recording of completed
sessions, not present-time liveness.

Do not expose 8081/admin without the host's private network/IP allowlist; the
legacy admin relies on that perimeter and does not implement its own password.
