# Nifty Scalper Bot — AWS Lightsail Operations

AWS Lightsail is the production deployment authority for this bot. The trading
engine runs as the `niftybot` systemd service on port `8080`.

## Secret-storage boundary

Production credentials are stored outside the Git checkout:

```text
/home/ubuntu/.config/niftybot/niftybot.env
```

The repository path `/home/ubuntu/nifty_scalper_bot/.env` is only a compatibility
symlink to that external file. Git pulls, resets, validation worktrees, and code
rollbacks therefore cannot overwrite broker or Telegram credentials.

Never commit a populated `.env` file. `.env.example` is the only repository
configuration template.

## One-time setup or migration

Run the setup script from the repository after preserving any existing local
configuration:

```bash
cd /home/ubuntu/nifty_scalper_bot
mkdir -p /home/ubuntu/.config/niftybot

if [ -f .env ] && [ ! -L .env ]; then
  cp -p .env /home/ubuntu/.config/niftybot/niftybot.env
  chmod 600 /home/ubuntu/.config/niftybot/niftybot.env
fi

bash deploy/lightsail_setup.sh
```

The setup script:

- migrates the historic in-repository `.env` file;
- installs the `niftybot` systemd unit;
- points `EnvironmentFile` to the external configuration;
- installs the validated release runner and two-minute update timer;
- keeps the canonical runtime entrypoint
  `nifty_scalper_bot.main:app`;
- preserves the existing dashboard and Caddy reverse proxy.

## Entering broker credentials

Use the browser dashboard or edit the external environment file directly:

```bash
nano /home/ubuntu/.config/niftybot/niftybot.env
```

At minimum, the engine requires a non-empty broker API key and API secret. LIVE
mode additionally requires a current access token and internally consistent flags:

```text
ENABLE_LIVE=true
EXECUTION_MODE=LIVE
```

Start in SHADOW mode until `/livez` reports `bot_loaded: true`.

## Deploying code

Do not run `git reset --hard origin/main` or a plain `git pull` as the deployment
procedure. Those commands do not install dependencies, restart the service, or
verify that the engine loaded.

Use the canonical release runner:

```bash
cd /home/ubuntu/nifty_scalper_bot
bash deploy/lightsail_release.sh --force
```

The runner performs a locked, staged deployment:

1. validates the external environment without printing secrets;
2. fetches `origin/main`;
3. creates a detached validation worktree;
4. compiles production source and runs focused architecture/execution tests;
5. installs the validated revision;
6. restarts `niftybot`;
7. requires `/livez` to report `bot_loaded: true`;
8. additionally requires `/readyz` in LIVE mode;
9. rolls code back if the new revision fails health validation.

The automatic update timer runs the same release runner every two minutes.

## Verification

```bash
sudo systemctl status niftybot --no-pager -l
curl -i http://127.0.0.1:8080/livez
curl -i http://127.0.0.1:8080/readyz
sudo journalctl -u niftybot -n 200 --no-pager
```

Healthy engine loading requires:

```json
{"status":"alive","bot_loaded":true}
```

A `200` response from `/livez` with `bot_loaded:false` means only the web process
is alive; the trading engine has not loaded. Treat that state as deployment
failure, not success.
