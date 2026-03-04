"""Unified execution engine test pipeline CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    """Args: cmd, env; Returns: int; Raises: None."""
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, check=False)
    return int(proc.returncode)


def _base_env(mode: str) -> dict[str, str]:
    """Args: mode; Returns: env mapping; Raises: None."""
    return {**os.environ, "PYTHONPATH": "src", "EXECUTION_MODE": mode}


def main() -> None:
    """Args: none; Returns: None; Raises: None."""
    parser = argparse.ArgumentParser(description="Run engine checks")
    parser.add_argument(
        "--mode", required=True, choices=["strategy", "stress", "replay", "paper"]
    )
    parser.add_argument("--file", default="backtests/data/sample_ticks.csv")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    replay_cmd = [
        sys.executable,
        "-m",
        "nifty_scalper_bot.testing.run_tick_replay",
        "--file",
        args.file,
        "--speed",
        "10",
    ]
    if args.deterministic:
        replay_cmd.append("--deterministic")

    if args.mode == "strategy":
        code = _run(
            [sys.executable, "-m", "nifty_scalper_bot.testing.strategy_test_runner"],
            env=_base_env("SIMULATION"),
        )
    elif args.mode == "stress":
        code = _run(
            [sys.executable, "-m", "nifty_scalper_bot.testing.execution_stress_tests"],
            env=_base_env("SIMULATION"),
        )
    elif args.mode == "replay":
        code = _run(replay_cmd, env=_base_env("SIMULATION"))
    else:
        code = _run(
            [sys.executable, "-m", "nifty_scalper_bot.testing.strategy_test_runner"],
            env=_base_env("PAPER"),
        )

    if args.mode == "stress" and code == 0:
        steps: list[tuple[list[str], dict[str, str]]] = [
            (
                [
                    sys.executable,
                    "-m",
                    "nifty_scalper_bot.testing.strategy_test_runner",
                ],
                _base_env("SIMULATION"),
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "nifty_scalper_bot.testing.execution_stress_tests",
                ],
                _base_env("SIMULATION"),
            ),
            (replay_cmd, _base_env("SIMULATION")),
            (
                [
                    sys.executable,
                    "-m",
                    "nifty_scalper_bot.testing.strategy_test_runner",
                ],
                _base_env("PAPER"),
            ),
            (
                [
                    sys.executable,
                    "-m",
                    "nifty_scalper_bot.testing.strategy_test_runner",
                ],
                _base_env("SHADOW"),
            ),
        ]
        for cmd, env in steps:
            code = _run(cmd, env=env)
            if code != 0:
                break

    sys.exit(code)


if __name__ == "__main__":
    main()
