"""Simple shared health state for orchestrator tests."""
from __future__ import annotations

from types import SimpleNamespace

STATE = SimpleNamespace(last_tick_ts=0.0)
