"""Regression: live simulation tests run without cross-suite class reloads."""

from __future__ import annotations

from pathlib import Path

from scripts import agent_check

ROOT = Path(__file__).resolve().parents[2]


def test_focused_validation_separates_simulated_runtime_tests() -> None:
    plan = agent_check.build(ROOT, ["src/nifty_scalper_bot/risk/risk_manager.py"])
    focus = [
        command for command in plan.commands if command.startswith("python -m pytest")
    ]
    assert agent_check.E2E_COMMAND in focus
    mixed = [command for command in focus if command != agent_check.E2E_COMMAND]
    assert mixed
    assert all("tests/e2e/live_sim" not in command for command in mixed)


def test_normal_validation_does_not_run_simulation_twice() -> None:
    plan = agent_check.build(ROOT, ["src/nifty_scalper_bot/risk/risk_manager.py"])
    assert plan.commands.count(agent_check.E2E_COMMAND) == 1
