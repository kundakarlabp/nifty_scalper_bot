from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDM = ROOT / "src/nifty_scalper_bot/data/market_data_manager.py"
RUNNER = ROOT / "src/nifty_scalper_bot/strategies/runner.py"
TEST = ROOT / "tests/runtime/test_readiness_history_integrity.py"
WORKFLOW = ROOT / ".github/workflows/one-shot-runtime-readiness-history-tdd.yml"
SCRIPT = Path(__file__).resolve()


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def replace_regex(path: Path, pattern: str, replacement: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"{label}: expected one source match, found {count}")
    path.write_text(updated, encoding="utf-8")
    print(f"applied: {label}", flush=True)


# Historical bar count must never promote a stale/missing live spot quote to ready.
replace_regex(
    MDM,
    r'(?m)^\s{12}spot_bar_ready = bars\.get\(spot, 0\) >= min_bars\n',
    "",
    "remove spot history promotion input",
)
replace_regex(
    MDM,
    r'(?m)^\s{12}if not spot_ready and spot_bar_ready:\n\s{16}spot_ready = True\n',
    "",
    "remove stale spot promotion",
)

# Reseeding canonical history is data hydration only; it must not claim runtime ownership.
replace_regex(
    RUNNER,
    r'(?m)^\s{16}self\._active_symbols\.add\(normalized_symbol\)\n'
    r'\s{16}self\._tracked_symbols\.add\(normalized_symbol\)\n',
    "",
    "remove reseed activation side effects",
)

# Fallback recovery must replace/deduplicate canonical history, not replay every cached row.
replace_regex(
    RUNNER,
    r'(?m)^\s{24}for bar_data in rows:\n'
    r'\s{28}self\.ingest_historical_bar\(bar_data\)\n'
    r'\s{28}total_bars \+= 1\n',
    '                        reseeded = self.reseed_history_from_bars(\n'
    '                            symbol,\n'
    '                            rows,\n'
    '                            source="runner_fallback",\n'
    '                            min_bars=target,\n'
    '                        )\n'
    '                        total_bars += int(reseeded or 0)\n',
    "use idempotent fallback reseed",
)

run("python", "-m", "black", str(MDM), str(RUNNER), str(TEST))
run("python", "-m", "pytest", str(TEST), "-q")
run("python", "-m", "compileall", "-q", str(MDM), str(RUNNER))
run(
    "python",
    "-m",
    "ruff",
    "check",
    str(MDM.relative_to(ROOT)),
    str(RUNNER.relative_to(ROOT)),
    str(TEST.relative_to(ROOT)),
)
run(
    "python",
    "-m",
    "black",
    "--check",
    str(MDM.relative_to(ROOT)),
    str(RUNNER.relative_to(ROOT)),
    str(TEST.relative_to(ROOT)),
)

run("git", "config", "user.name", "github-actions[bot]")
run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
run("git", "rm", "-f", str(WORKFLOW.relative_to(ROOT)), str(SCRIPT.relative_to(ROOT)))
run(
    "git",
    "add",
    str(MDM.relative_to(ROOT)),
    str(RUNNER.relative_to(ROOT)),
    str(TEST.relative_to(ROOT)),
)
run("git", "commit", "-m", "fix(runtime): preserve readiness and history ownership")
run("git", "push", "origin", "HEAD")
