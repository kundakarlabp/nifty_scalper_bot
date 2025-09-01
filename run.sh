#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for platforms expecting a run.sh entrypoint.
# Delegates to manage_bot.sh to preserve existing behavior.
exec bash manage_bot.sh run
