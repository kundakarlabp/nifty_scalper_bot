#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for platforms expecting a run.sh entrypoint.
# Delegates to manage_bot.sh to preserve existing behavior while ensuring
# unbuffered logging is enabled for downstream processes.

if [[ $# -eq 0 ]]; then
  set -- run
fi

export PYTHONUNBUFFERED=1
exec bash manage_bot.sh "$@"
