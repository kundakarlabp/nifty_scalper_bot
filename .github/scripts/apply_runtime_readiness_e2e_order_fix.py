from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/e2e/live_sim/test_live_runtime_startup_contract.py"
WORKFLOW = ROOT / ".github/workflows/one-shot-runtime-readiness-e2e-order.yml"
SCRIPT = Path(__file__).resolve()


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


text = TEST.read_text(encoding="utf-8")
text, removed = re.subn(
    r"(?m)^(\s*)ctx\.websocket_manager\.connect\(\)\n\1ctx\.strategy_runner\.start\(\)\n",
    r"\1ctx.websocket_manager.connect()\n",
    text,
    count=1,
)
if removed != 1:
    raise SystemExit(f"remove premature runner start: expected one match, found {removed}")
text, inserted = re.subn(
    r"(?m)^(\s*)_pump_runtime\(loop, ctx, iterations=20\)\n\n(\s*)assert ctx\.live_orders_armed is False",
    r"\1_pump_runtime(loop, ctx, iterations=20)\n\1# Runtime ownership is established by explicit registration above; start only after it.\n\1ctx.strategy_runner.start()\n\1_pump_runtime(loop, ctx, iterations=2)\n\n\2assert ctx.live_orders_armed is False",
    text,
    count=1,
)
if inserted != 1:
    raise SystemExit(f"insert runner start after registration: expected one match, found {inserted}")
TEST.write_text(text, encoding="utf-8")

run(
    "python",
    "-m",
    "pytest",
    "-q",
    "-m",
    "live_runtime_e2e",
    "tests/e2e/live_sim/test_live_runtime_startup_contract.py::test_live_runtime_bullish_spot_future_selects_ce_and_exits_target",
)
run("git", "config", "user.name", "github-actions[bot]")
run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
run("git", "rm", "-f", str(WORKFLOW.relative_to(ROOT)), str(SCRIPT.relative_to(ROOT)))
run("git", "add", str(TEST.relative_to(ROOT)))
run("git", "commit", "-m", "test(e2e): register runtime symbols before runner start")
run("git", "push", "origin", "HEAD")
