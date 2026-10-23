"""Regression: live simulation tests run without cross-suite class reloads."""

from __future__ import annotations

from pathlib import Path

from scripts.agent_check import build, E2E_COMMAND


ROOT = Path(__file__).resolve().parents[2]


def test_focused_validation_separates_simulated_runtime_tests() -> None:
    plan = build(ROOT, ["src/nifty_scalper_bot/risk/risk_manager.py"])
    focus = [
        command for command in plan.commands if command.startswith("python -m pytest")
    ]
    assert E2E_COMMAND in focus
    mixed = [command for command in focus if command != E2E_COMMAND]
    assert mixed
    assert all("tests/e2e/live_sim" not in command for command in mixed)


def test_normal_validation_does_not_run_simulation_twice() -> None:
    plan = build(ROOT, ["src/nifty_scalper_bot/risk/risk_manager.py"])
    assert plan.commands.count(E2E_COMMAND) == 1
