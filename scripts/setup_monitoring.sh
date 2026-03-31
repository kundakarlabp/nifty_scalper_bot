#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MONITORING_DIR="${PROJECT_ROOT}/ops/monitoring"
PROMETHEUS_DATA="${PROJECT_ROOT}/data/prometheus"
GRAFANA_DATA="${PROJECT_ROOT}/data/grafana"

mkdir -p "${PROMETHEUS_DATA}" "${GRAFANA_DATA}" "${MONITORING_DIR}/prometheus_multiproc"

cat <<EOF
Monitoring directories initialised.
Prometheus config: ${MONITORING_DIR}/prometheus.yml
Grafana provisioning: ${MONITORING_DIR}/grafana
Prometheus data dir: ${PROMETHEUS_DATA}
Grafana data dir: ${GRAFANA_DATA}
EOF

if docker compose version >/dev/null 2>&1; then
  echo "Restarting monitoring stack..."
  docker compose -f "${PROJECT_ROOT}/docker-compose.yml" up -d prometheus grafana
else
  echo "docker compose command not found; skipping automatic restart." >&2
fi
