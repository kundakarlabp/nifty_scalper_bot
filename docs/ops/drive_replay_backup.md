# NIFTY replay archive: private Google Drive backup

## Owner and purpose

Canonical market evidence lives on AWS Lightsail in `DATA_DIR/replay_archive`,
not in GitHub, the Supabase daily-log table or ChatGPT's Google Drive connector.
The GitHub repository holds bot source code, tests and revisions. Google Drive
is a secondary, immutable off-site copy of *completed* sessions.

`storage/replay_compaction.py` remains the source of verified, lossless gzip
archives. Its existing hourly, off-market background task invokes the optional
`storage/replay_drive_backup.py` copier when `REPLAY_GDRIVE_REMOTE` is set.
No new trading-data ingestor, order path, listener, or independent schedule exists.

## One-time authorization on the production host

ChatGPT's connected Drive OAuth is not a credential for the Lightsail service.
A separate, securely stored Google Drive authorization is required once.

1. Install a supported `rclone` version on the Lightsail host.
2. As the Linux user that runs the `niftybot` service, run `rclone config`.
   Configure a `drive` remote named `niftybackup`, authorize the intended
   Google account through Google's browser/OAuth process, and set its advanced
   `root_folder_id` to the **existing private archive folder ID**. Use a
   Google OAuth client per current rclone guidance where required.
3. Confirm read/write access with `rclone lsd niftybackup:`, and ensure the
   rclone configuration is readable only by the bot service user. Never
   commit OAuth tokens, rclone config, Google client secrets or session files.
4. In the protected **host** environment, set:
   `REPLAY_GDRIVE_REMOTE=niftybackup:`
   Then use the canonical broker-safe release procedure. Do not manually
   restart an armed bot to activate archival.

The remote is rooted at the archive folder, not at the user's whole Drive.
Only the remote name, not any OAuth token, is read from the bot environment.
With an unset remote, all cloud-upload behavior is disabled.

## Daily lifecycle and verified contents

- Capture every delivered normalized, subscribed WebSocket observation into
  `YYYY-MM-DD.jsonl` with its `<run_id>.integrity.json` sidecars.
- After the trading day, the existing research job writes markout and
  fidelity JSON if its evidence checks pass or fail.
- **After midnight IST**, the existing off-market compactor converts prior-day
  JSONL into verified `YYYY-MM-DD.jsonl.gz`. It does not touch the current
  day, and validates the uncompressed SHA before deleting its local plain copy.
- At the next off-market maintenance cycle, the backup job validates the
  compressed tape with `iter_session`, and uploads that file, matching run
  integrity sidecars, optional research JSON, and a SHA-256 manifest to
  `niftybackup:YYYY/YYYY-MM-DD/`.
- Each file is verified against Google Drive's remote MD5 through rclone.
  MD5 is used *only for remote byte comparison*; local SHA-256 manifests retain
  the archival integrity record. A local `replay_drive_backups/`
  receipt is created **only after all files verify**.
- No uploaded or local source is automatically deleted by this uploader.
  Existing receipt skips prevent duplicate uploads. A failed transfer is retried
  off-market, without claiming success.

The copied tape provides replay evidence for **subscribed contracts**. It
does not prove the entire NSE option chain, complete exchange packet delivery,
broker fills, or execution profitability. Preserve the integrity sidecars
alongside the tape; replay validation requires them.

## Monitoring and limits

A successful daily upload emits `REPLAY_DRIVE_BACKUP day=... status=uploaded_verified`.
A failed attempt emits `REPLAY_DRIVE_BACKUP_FAILED error_type=...`.
No market-hour disk compression or Google Drive traffic is added.

The source tape can exceed 400 MiB/day before compression. Monitor Google
Drive capacity and plan storage upgrades/retention; backup stops visibly on
quota errors rather than silently sampling or overwriting raw data. If a
session has unresolved integrity gaps, the uploader refuses the tape.
Archive availability and integrity do **not** waive the independent
backtest-fidelity and cost-model gates.
