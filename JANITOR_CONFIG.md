# Repo Janitor – What it flags
- **Exact duplicates** (same content hash)
- **Near-duplicates (.py)** – comments/whitespace ignored
- **Big artifacts** – >5 MB (config JANITOR_BIG_MB)
- **Orphaned modules (.py)** – not reachable from entry points/tests

## Safe cleanup policy
- We **archive** instead of deleting: `archive/` dir via `git mv`
- Review `JANITOR_REPORT.json` before moving anything
