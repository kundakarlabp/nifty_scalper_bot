#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKUP_ROOT="${PROJECT_ROOT}/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

mkdir -p "${TARGET_DIR}"

rsync -a "${PROJECT_ROOT}/logs/" "${TARGET_DIR}/logs/" --exclude "*.tmp" || true
rsync -a "${PROJECT_ROOT}/data/" "${TARGET_DIR}/data/" || true
rsync -a "${PROJECT_ROOT}/config/" "${TARGET_DIR}/config/"

find "${BACKUP_ROOT}" -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +

echo "Backup complete: ${TARGET_DIR}"
