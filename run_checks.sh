#!/usr/bin/env bash
# run_checks.sh — setup, lint (if available), test, and run backtester
set -euo pipefail

# ---------------------------
# Resolve repo root & cd there
# ---------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== run_checks.sh: starting in $(pwd) ==="

# ---------------------------
# Environment / Python setup
# ---------------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_NO_CACHE_DIR=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
echo "PYTHONPATH=${PYTHONPATH}"

# Use a local venv unless running inside CI (Docker image may already have deps)
if [[ -z "${CI:-}" ]]; then
  if [[ ! -d ".venv" ]]; then
    echo "--- Creating virtualenv (.venv) ---"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="python"
  PIP_BIN="python -m pip"
else
  PYTHON_BIN="python3"
  PIP_BIN="python3 -m pip"
fi

echo "--- Python version ---"
$PYTHON_BIN --version

echo "--- Upgrading pip ---"
$PIP_BIN install --upgrade pip

# ---------------------------
# Dependencies
# ---------------------------
if [[ -f "requirements.txt" ]]; then
  echo "--- Installing requirements ---"
  $PIP_BIN install -r requirements.txt
else
  echo "⚠ requirements.txt not found; skipping."
fi

# Optional: dev requirements (ignored if file absent)
if [[ -f "requirements-dev.txt" ]]; then
  echo "--- Installing dev requirements ---"
  $PIP_BIN install -r requirements-dev.txt
fi

# ---------------------------
# Static checks (best-effort)
# ---------------------------
if $PYTHON_BIN -m pip show ruff >/dev/null 2>&1; then
  echo "--- Ruff lint (if configured) ---"
  ruff --version || true
  # Lint src/ and tests/ if present
  if [[ -d "src" ]]; then ruff check src || true; fi
  if [[ -d "tests" ]]; then ruff check tests || true; fi
else
  echo "ℹ Ruff not installed; skipping lint."
fi

if $PYTHON_BIN -m pip show mypy >/dev/null 2>&1; then
  echo "--- mypy type check (best-effort) ---"
  if [[ -d "src" ]]; then mypy src || true; fi
else
  echo "ℹ mypy not installed; skipping type check."
fi

# ---------------------------
# Unit tests
# ---------------------------
if [[ -d "tests" ]]; then
  echo "--- Running pytest ---"
  # -q quiet; -x fail fast; add --maxfail=1 to bail on first failure
  if $PYTHON_BIN -m pip show pytest >/dev/null 2>&1; then
    pytest -q -x
  else
    echo "❌ pytest not installed but tests/ exists. Install pytest or remove tests/"; exit 1
  fi
else
  echo "ℹ tests/ directory not found; skipping pytest."
fi

# ---------------------------
# Backtest (integration smoke)
# ---------------------------
BACKTEST="tests/true_backtest_dynamic.py"
if [[ -f "$BACKTEST" ]]; then
  echo "--- Running backtester: $BACKTEST ---"
  # Provide sane defaults for time filters if your script reads them
  export TIME_FILTER_START="${TIME_FILTER_START:-09:15}"
  export TIME_FILTER_END="${TIME_FILTER_END:-15:30}"
  $PYTHON_BIN "$BACKTEST"
else
  echo "ℹ $BACKTEST not found; skipping backtest run."
fi

echo "✅ All checks completed successfully."
