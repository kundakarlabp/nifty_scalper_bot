#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${OUTDIR:-$ROOT/data}"
DAYS="${DAYS:-365}"
VERBOSE="${VERBOSE:-}"

mkdir -p "$OUTDIR"

if [[ -n "${VERBOSE}" ]]; then
  python3 "$ROOT/scripts/sync_nifty_options.py" --days "$DAYS" --outdir "$OUTDIR" --verbose
else
  python3 "$ROOT/scripts/sync_nifty_options.py" --days "$DAYS" --outdir "$OUTDIR"
fi
