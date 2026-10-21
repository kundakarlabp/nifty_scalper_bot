# Nifty Scalper Bot — AWS Lightsail Operations

AWS Lightsail is the production deployment authority. The trading engine runs as
`niftybot` on port `8080`; the lightweight admin-control service runs as
`niftybot-admin` on port `8081`; the separate read-only Streamlit operations
console runs as `niftybot-streamlit` on port `8501`. Admin controls and the
read-only review page are isolated processes; neither overwrites unspecified
external environment settings.

## Secret-storage boundary

Production credentials are stored outside the Git checkout:

```text
/home/ubuntu/.config/niftybot/niftybot.env
```

The repository path `/home/ubuntu/nifty_scalper_bot/.env` is only a compatibility
symlink to that external file. Git pulls, resets, validation worktrees, and code
rollbacks therefore cannot overwrite broker or Telegram credentials.

Never commit a populated `.env` file. `.env.example` is the repository template.

## One-time setup or migration

```bash
cd /home/ubuntu/nifty_scalper_bot
mkdir -p /home/ubuntu/.config/niftybot

if [ -f .env ] && [ ! -L .env ]; then
  cp -p .env /home/ubuntu/.config/niftybot/niftybot.env
  chmod 600 /home/ubuntu/.config/niftybot/niftybot.env
fi

bash deploy/lightsail_setup.sh
```

For a previously installed instance that only needs the isolated Streamlit
console installed, run `bash deploy/scripts/install_streamlit_console.sh`
after verifying the current engine release. Routine code updates must use the
validated release runner below, not a separate ad-hoc deployment path.

The setup script:

- migrates the historic in-repository `.env` file;
- installs the `niftybot` engine service;
- installs the isolated `.streamlit-venv` and `niftybot-streamlit` console service;
- points the engine to the external environment file;
- installs the validated release runner and two-minute update timer;
- keeps `nifty_scalper_bot.main:app` as the canonical engine entrypoint;
- restarts both engine and console after validated GitHub updates.

Allow TCP `443` and TCP `8501` in the Lightsail firewall. Port `8080` is needed
only when direct admin fallback access is required.

## Console URLs

```text
Operations console: http://LIGHTSAIL_STATIC_IP:8501
Admin dashboard:    https://LIGHTSAIL_STATIC_IP/admin
Admin fallback:     http://LIGHTSAIL_STATIC_IP:8080/admin
```

The operations console is read-only. Historical journal scans run only after the
**Generate download** button is pressed. Live fragments never scan historical logs.

## Entering broker credentials

Use the admin dashboard or edit the external environment file:

```bash
nano /home/ubuntu/.config/niftybot/niftybot.env
```

LIVE mode requires a current access token and consistent flags:

```text
ENABLE_LIVE=true
EXECUTION_MODE=LIVE
```

Start in SHADOW mode until `/livez` reports `bot_loaded: true`.

## Deploying code

Do not use a plain `git pull` or `git reset --hard origin/main` as the deployment
procedure. Use the validated release runner:

```bash
cd /home/ubuntu/nifty_scalper_bot
bash deploy/lightsail_release.sh --force
```

The runner:

1. validates the external environment without printing credentials;
2. fetches `origin/main` into a detached validation worktree;
3. compiles engine and dashboard source;
4. validates all Lightsail shell scripts;
5. runs focused architecture, execution, dashboard and export tests;
6. installs the validated engine revision;
7. restarts `niftybot` and requires `bot_loaded: true`;
8. restarts and health-checks `niftybot-streamlit`;
9. rolls code back if the trading engine fails health validation;
10. reports console failure separately so a healthy trading engine is not disrupted.

The automatic update timer fetches `origin/main` every two minutes. PR commits
are not deployed until merged into `main` (unreviewed PR heads must never
replace production). When a new main SHA is available, the updater validates
the candidate in a detached worktree before attempting a release.

Automatic restarts are **deferred between 09:15 and 15:45 IST** on weekdays,
regardless of a momentarily flat account: the runner could otherwise enter a
new trade between broker probes and the restart. At all other times, including
08:00–09:14 pre-market and after the close, a release still requires a
reconciled, **disarmed** engine and two broker-confirmed snapshots with **no**
positions or outstanding orders. If the bot is in its canonical off-hours
**QUIET** mode (trading stack unloaded), the updater verifies that mode through
`/livez` and still requires the same two broker snapshots. If the broker
cannot prove flatness, a session
token has expired, tests fail, or a trade remains open after close, the pending
SHA stays queued; the same timer retries without forcing a restart. Candidate
checkout and engine restart each have a fresh safety check. Never reset a risk
limit or cancel protective orders to accelerate deployment.

See `data/auto_update_status.json` or the Admin status page for
`deferred_market_window`, `deferred_unsafe_recovery`, validation failures,
and `deployed` status. A deferred release is not proof that the bot is down.
The **Update from GitHub** button requests the same safe updater; it does not
bypass the protection window. If the timer is disabled or consistently failing,
restore `niftybot-autodeploy.timer` through the existing setup; a Git commit
alone cannot start an inactive host service.

Use this command directly for manual releases; obsolete `ops/redeploy.sh` and
`deploy/scripts/update_and_deploy.sh` forwarding shortcuts are retired.
Existing host-side shortcuts must be updated to call the canonical runner.

## Log exports

The console supports:

- current filtered event-buffer CSV;
- exact IST market-session actionable-event CSV;
- selective event-type, text and trade-lifecycle filtering;
- full redacted service-log TXT;
- bounded previews of 100, 250 or 500 rows.

IST windows are converted to absolute Unix epochs before `journalctl` is called,
so morning reports remain correct even though the Lightsail host runs in UTC.
Exports are limited to the newest 24 MB and a 35-second query time to prevent
market-hour memory or CPU spikes.

## Verification

```bash
sudo systemctl status niftybot --no-pager -l
sudo systemctl status niftybot-streamlit --no-pager -l
curl -sS http://127.0.0.1:8080/livez
curl -I http://127.0.0.1:8501
sudo journalctl -u niftybot -n 200 --no-pager
sudo journalctl -u niftybot-streamlit -n 100 --no-pager
```

Healthy engine loading requires:

```json
{"status":"alive","bot_loaded":true}
```

A `200` response from `/livez` with `bot_loaded:false` means only the HTTP process
is alive; the trading engine has not loaded.
