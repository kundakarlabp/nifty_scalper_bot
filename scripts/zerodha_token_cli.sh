#!/usr/bin/env bash
set -euo pipefail
python3 "$(dirname "$0")/../zerodha_token_cli.py" "$@"
