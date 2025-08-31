#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
pip install -e . || true

if [ -f .pre-commit-config.yaml ]; then
  pre-commit run --all-files
else
  echo "No .pre-commit-config.yaml; skipping."
fi

pytest -q
# Optional slow smoke (non-fatal)
pytest -q -m slow || true

echo "✅ All checks completed successfully."
