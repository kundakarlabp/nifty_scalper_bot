#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required" >&2
  exit 1
fi

echo "Building images..."
docker compose -f "${COMPOSE_FILE}" build --pull

echo "Starting stack..."
docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans

echo "Services running. Use 'docker compose -f ${COMPOSE_FILE} ps' to inspect."
