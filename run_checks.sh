#!/bin/bash
set -e

echo "--- Setting up environment ---"
export PYTHONPATH=$PYTHONPATH:.
echo "PYTHONPATH is $PYTHONPATH"

echo "--- Installing dependencies ---"
pip install -r requirements.txt

echo "--- Running pytest ---"
pytest

echo "--- Running backtester ---"
python3 tests/true_backtest_dynamic.py
