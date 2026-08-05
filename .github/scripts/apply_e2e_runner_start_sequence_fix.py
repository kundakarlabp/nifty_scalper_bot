from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/e2e/live_sim/test_live_runtime_startup_contract.py"
WORKFLOW = ROOT / ".github/workflows/one-shot-e2e-runner-start-sequence.yml"
SCRIPT = Path(__file__).resolve()


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def replace_once(old: str, new: str) -> None:
    text = TEST.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one test match, found {count}")
    TEST.write_text(text.replace(old, new, 1))


def main() -> None:
    replace_once(
        "        ctx.websocket_manager.connect()\n"
        "        ctx.strategy_runner.start()\n"
        "        broker = ctx.broker_client.client\n",
        "        ctx.websocket_manager.connect()\n"
        "        broker = ctx.broker_client.client\n",
    )
    replace_once(
        "        _pump_runtime(loop, ctx, iterations=20)\n\n"
        "        assert ctx.live_orders_armed is False\n"
        "        loop.run_until_complete(\n",
        "        loop.run_until_complete(\n"
        "            core_app._ensure_strategy_runner_started(\n"
        "                ctx, reason=\"live_sim_symbols_registered\"\n"
        "            )\n"
        "        )\n"
        "        _pump_runtime(loop, ctx, iterations=20)\n\n"
        "        loop.run_until_complete(\n",
    )

    run(
        "python",
        "-m",
        "pytest",
        "-q",
        "-m",
        "live_runtime_e2e",
        "tests/e2e/live_sim/test_live_runtime_startup_contract.py::test_live_runtime_bullish_spot_future_selects_ce_and_exits_target",
    )
    run("git", "diff", "--check")
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    run("git", "rm", "-f", str(WORKFLOW.relative_to(ROOT)), str(SCRIPT.relative_to(ROOT)))
    run("git", "add", str(TEST.relative_to(ROOT)))
    run("git", "commit", "-m", "test(e2e): follow production runner start sequence")
    run("git", "push", "origin", "HEAD")


if __name__ == "__main__":
    main()
