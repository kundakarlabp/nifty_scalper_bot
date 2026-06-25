# AWS Lightsail Release Validation

The host updater validates every changed `main` revision before activating it.

## Candidate checks

1. Check out the candidate commit in an isolated git worktree.
2. Compile the production Python source.
3. Run the focused architecture, execution-facade and end-to-end lifecycle tests that exist in the candidate revision.
4. Leave the running checkout and service untouched when validation fails.

## Activation and rollback

After validation, the updater resets the active checkout to the candidate revision, reinstalls the editable package and restarts `niftybot.service`. It then polls the local `/livez` endpoint. Failure to recover causes an automatic reset to the previous commit, package reinstall and service restart.

The status of each attempt is written atomically to `data/auto_update_status.json` for the operations dashboard.

The repository CI remains responsible for the complete test suite; the host performs a focused release gate to avoid excessive production load.
