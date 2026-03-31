"""Unified trading engine test pipeline CLI."""

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


def main() -> None:
    """Args: none; Returns: None; Raises: None."""
    parser = argparse.ArgumentParser(description='Run deterministic engine checks')
    parser.add_argument('--mode', required=True, choices=['replay', 'paper', 'stress'])
    parser.add_argument('--file', default='backtests/data/sample_ticks.csv')
    args = parser.parse_args()

    if args.mode == 'replay':
        code = _run(
            [
                sys.executable,
                '-m',
                'nifty_scalper_bot.testing.run_tick_replay',
                '--file',
                args.file,
                '--speed',
                '10',
            ],
            env={**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'SIMULATION'},
        )
    elif args.mode == 'paper':
        code = _run(
            [sys.executable, '-m', 'nifty_scalper_bot.testing.strategy_test_runner'],
            env={**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'PAPER'},
        )
    else:
        steps = [
            [
                sys.executable,
                '-m',
                'nifty_scalper_bot.testing.run_tick_replay',
                '--file',
                args.file,
                '--speed',
                '10',
            ],
            [sys.executable, '-m', 'nifty_scalper_bot.testing.execution_stress_tests'],
            [sys.executable, '-m', 'nifty_scalper_bot.testing.strategy_test_runner'],
            [sys.executable, '-m', 'nifty_scalper_bot.testing.strategy_test_runner'],
        ]
        envs = [
            {**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'SIMULATION'},
            {**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'SIMULATION'},
            {**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'PAPER'},
            {**os.environ, 'PYTHONPATH': 'src', 'EXECUTION_MODE': 'SHADOW'},
        ]
        code = 0
        for cmd, env in zip(steps, envs):
            code = _run(cmd, env=env)
            if code != 0:
                break

    sys.exit(code)


if __name__ == '__main__':
    main()
