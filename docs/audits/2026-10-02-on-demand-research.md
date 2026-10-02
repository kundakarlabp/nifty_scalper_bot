# On-demand research using the existing AWS updater

The validated Lightsail updater polls `deploy/research_request.json` on healthy
current revisions and after successful releases. A new ID requests one job;
repeated polls of the same ID return its existing status. The admin dashboard
uses the same launcher. No shell commands, paths, broker credentials or strategy
parameters are accepted from a request. The job runs separately with live flags
disabled and does not restart or reconfigure trading.

The worker uses the existing authenticated operator environment, current
runtime option basket, canonical active-future resolver and Kite history
exporter. Only completed calendar dates are requested, for at most 90 days.
It also calls the existing canonical completed-trade analyzer with actual-cost
requirements. Reports are stored in `data/research/<id>/`, with HTTP status and
latest analysis at `/admin/research/status` and `/admin/research/report`.
The existing `/trading/status` diagnostic also includes `research_job`, so the
unchanged read-only AppDeploy relay can retrieve job status from ChatGPT.

## Evidence boundary

This change automates research prerequisites, **not a complete current-bot
historical backtest**. The existing standalone backtest has an RSI bridge and
the nightly script defaults to a demonstration momentum strategy. Neither is
the live ORBPro/SMC/VWAPPro pipeline. The worker explicitly reports
`current_bot_offline_replay_adapter_unavailable`, with
`backtest_completed=false`, after collection. It never substitutes demonstration
PnL or synthetic bid/ask depth. Active-contract minute history cannot recover
expired option contracts or establish historical ATM selection before capture.

## Operations and rollback

The included request is one automatically picked up 30-day collection attempt
when this revision is validated and deployed. Authentication failure or missing
active basket leaves a failed status, never success. The immutable request ID
prevents silent repeated retries. A fresh ID explicitly retries. The OS lock is
held by the detached worker until exit; concurrent starts return busy. Removing
the request manifest stops GitHub-initiated requests; dashboard requests remain
explicit. No AppDeploy relay settings or version are changed.

Deployment and the real broker job must be observed separately from local tests.

## Observed deployment lifecycle correction

The first deployed request remained queued. The auto-updater is a systemd
oneshot, whose default control-group cleanup can terminate a detached worker
when its parent exits. CLI polling now waits for the fixed worker (maximum
30 minutes), then reports its actual terminal status; it records timeout or
unexpected child exit as failure. Dashboard callers still launch separately
from their persistent service. Request `research-20261002-02` explicitly retries
collection after this correction. No systemd permissions are expanded.

The second request reached the worker but returned a generic `ValueError`.
Stage reporting now preserves collected history when ledger validation fails,
classifies only fixed known errors without exposing exception payloads, and
waits for the selected basket during background engine startup. Missing verified
ledger costs remain a blocker; the job does not replace them with estimates.
Request `research-20261002-03` retries with these observable failure boundaries.
