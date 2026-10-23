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

Never commit a populated `.env` file. `.env.example` is an example
for development, **not** the source of production strategy defaults. The
runtime's GitHub-owned configuration modules own those defaults.

Fresh Lightsail installations now create only a minimal secret/operator file
(`ENABLE_LIVE=false`, `EXECUTION_MODE=SHADOW`); neither setup nor Python's
config normalizer silently authorizes real orders. To enable an existing LIVE
installation, both canonical flags must already be explicitly set and agree.
Inconsistent settings fail closed and do not rewrite the host's protected file.

Existing populated host environment files are preserved to avoid unexpected
changes to trading behavior. A subsequent migration of legacy strategy
overrides must compare *effective* values, not blindly delete keys while
trading. Do not run the full setup script merely to edit credentials.

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

- preserves existing external credentials, migrating a legacy `.env` only once;
- creates a minimal SHADOW-only host env on fresh installs, with no strategy defaults;
- installs the `niftybot` engine service;
- installs the isolated `.streamlit-venv` and `niftybot-streamlit` console service;
- points the engine to the external environment file;
- installs the validated release runner and 30-second non-overlapping update timer;
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

The automatic release timer checks `origin/main` 30 seconds after its previous run finishes. PR commits
are not deployed until merged into `main` (unreviewed PR heads must never
replace production). After a merge, the updater additionally requires a
**completed successful push-triggered GitHub CI run for the exact main SHA**
before candidate validation or checkout mutation. It uses the public GitHub
Actions API without storing a token. Pending CI produces `deferred_ci`; failed
CI produces `ci_failed`; an unreachable/rate-limited/invalid CI API response
produces `ci_unavailable`. All three leave the current runtime untouched and
retry on the next timer invocation. An already deployed revision does not need
a new CI request to remain healthy or restart under the existing safety policy.

When a new CI-approved main SHA is available, the updater validates the
candidate in a detached worktree before attempting a release. The 30-second
interval is the polling interval, **not a guaranteed deployment deadline**:
CI duration, broker exposure/arming, and transient network failures can defer
a release without compromising live position protection.

There is **no market-hours restart blackout**. After full candidate
validation the updater exclusively locks the live BUY order-submission path
and publishes an entry hold acknowledged by the running engine. Protective
SELL exits remain available. Two independently fresh Kite snapshots must
show zero net/day positions and **only terminal orders**, with fresh reconciled
engine health and no entry arming. A pending/unknown order, open position,
unavailable API, failed reconciliation or missing hold acknowledgement
defers the deployment **automatically** until the next timer check. No
risk budget, stop order, or live position is modified to facilitate release.
The lock and hold cover checkout mutation, reinstall and engine restart;
on normal completion the hold is released, allowing the new engine to arm
from its own readiness policy.

For the initial upgrade from an older engine without entry-hold support,
the existing engine must be independently QUIET/disarmed and broker-flat.
Do not bypass this one-time bootstrap guard while LIVE-armed. An engine
that cannot acknowledge the hold is **not** automatically treated as safe
to restart. CI and deploy health checks remain separate; neither PR merge
nor a green GitHub workflow proves that production is running the new SHA.

See `data/auto_update_status.json` or the Admin status page for
`deferred_broker_exposure`, `deferred_unsafe_recovery`, validation failures,
and `deployed` status. A deferred release is not proof that the bot is down.
The **Update from GitHub** button requests the same safe updater; it does not
bypass broker-state and engine-arming protection. If the timer is disabled or consistently failing,
restore `niftybot-autodeploy.timer` through the existing setup; a Git commit
alone cannot start an inactive host service.

Use this command directly for manual releases; obsolete `ops/redeploy.sh` and
`deploy/scripts/update_and_deploy.sh` forwarding shortcuts are retired.
Existing host-side shortcuts must be updated to call the canonical runner.

## Recover a stale installed validator (one-time bootstrap)

If the systemd timer keeps reporting `validation_failed` because the installed
release script attempts to compile a helper that was legitimately deleted on
`main` (for example `dashboard/event_buffer.py`), **do not recreate the
retired helper**, run `git reset --hard`, or force-restart the engine.

The installed release script validates a detached candidate **before** changing
the checkout, so it cannot deploy its own validation fix. A one-time,
operator-approved bootstrap can run the **GitHub-reviewed, CI-validated** release
runner directly from its pinned `main` commit without changing the current
checkout first. This is an exceptional host recovery procedure, not a second
deployment mechanism:

1. Confirm the exact main SHA has passed final GitHub CI and is authorized.
2. Stop only `niftybot-autodeploy.timer` temporarily to prevent concurrent
   invocations; do not stop or restart the trading engine.
3. `git fetch origin main` without changing HEAD, write
   `git show <validated-sha>:deploy/lightsail_release.sh` to a temporary file,
   and check `bash -n`. Never execute an unreviewed PR runner.
4. Invoke the temporary validated runner as the existing `ubuntu` service
   user with `--auto`. It still validates the candidate and checks
   broker-flat/disarmed safety before mutating the checkout or restarting.
5. Restore the timer even on failure. Verify the running `/releasez` SHA,
   all service health endpoints, and the timer. A deferred/failed release is
   not a successful deployment and must be investigated from its status and
   journal.

Routine updates must continue through the single canonical
`deploy/lightsail_release.sh` installed in the repo; the dashboard validator
compiles the actual candidate dashboard tree rather than pinning retired
internal helper filenames.

## Fast code-only releases

The single validated Lightsail release runner now reuses an already-installed
editable Python package when the committed packaging and dependency manifests
are unchanged. Ordinary changes to `src/` need only the guarded checkout,
service restart and health verification; they do not mutate installed Python
dependencies. A missing editable install or changes to `pyproject.toml`,
setup files, requirements or lockfiles retain the full `pip install -e .`
path. When a dependency installation was attempted, rollback restores the
previous dependency environment as before. All GitHub CI, broker-flat
verification, entry hold, SHA validation and systemd rollback guards remain.

This is a deployment-speed optimization, **not** a second deployment system.
A future fully versioned/atomic virtual environment migration would require
separate systemd and rollback regression tests before it could replace the
existing service path.

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
