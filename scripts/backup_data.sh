#!/usr/bin/env bash
# Back up runtime logs and data. Production credentials remain external to Git.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKUP_ROOT="${PROJECT_ROOT}/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

# Root config/ was retired; never copy external credential storage into backups.
if [ ! -d "${PROJECT_ROOT}/logs" ] && [ ! -d "${PROJECT_ROOT}/data" ]; then
  printf 'ERROR: no logs/ or data/ directory is available to back up\n' >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
for name in logs data; do
  if [ -d "${PROJECT_ROOT}/${name}" ]; then
    rsync -a --exclude='*.tmp' "${PROJECT_ROOT}/${name}/" "${TARGET_DIR}/${name}/"
  fi
done

# Rotate only child backup directories, never the backup root itself.
find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf -- {} +

printf 'Backup complete: %s\n' "${TARGET_DIR}"
