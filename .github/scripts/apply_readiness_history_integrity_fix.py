from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/nifty_scalper_bot/strategies/runner.py"
MDM = ROOT / "src/nifty_scalper_bot/data/market_data_manager.py"
TEST = ROOT / "tests/strategies/test_runtime_history_activation_integrity.py"
WORKFLOW = ROOT / ".github/workflows/one-shot-readiness-history-integrity.yml"
SCRIPT = Path(__file__).resolve()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    return result


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}: found {count}")
    path.write_text(text.replace(old, new, 1))


def replace_function(
    path: Path, start_marker: str, end_marker: str, new_function: str
) -> None:
    text = path.read_text()
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"function start not found in {path}: {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"function end not found in {path}: {end_marker!r}")
    path.write_text(text[:start] + new_function + text[end:])


def prove_red() -> None:
    result = run(
        "python",
        "-m",
        "pytest",
        "-q",
        str(TEST.relative_to(ROOT)),
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError("red phase unexpectedly passed")
    print("RED_PHASE_CONFIRMED", flush=True)


def apply_fix() -> None:
    replace_once(
        RUNNER,
        '''                if one_minute_bars:\n                    self._last_bar_ts[normalized_symbol] = one_minute_bars[-1].start\n                self._active_symbols.add(normalized_symbol)\n                self._tracked_symbols.add(normalized_symbol)\n                self._data_phase.setdefault(normalized_symbol, "HYDRATION")\n''',
        '''                if one_minute_bars:\n                    self._last_bar_ts[normalized_symbol] = one_minute_bars[-1].start\n                self._data_phase.setdefault(normalized_symbol, "HYDRATION")\n''',
    )

    replace_function(
        RUNNER,
        "    async def _backfill_history(self) -> None:\n",
        "    def _hydrate_missing_bars(",
        '''    async def _backfill_history(self) -> None:\n        """Restore canonical history for active symbols still short after app hydration."""\n        total_bars = 0\n\n        try:\n            with self._lock:\n                targets = list(self._active_symbols)\n\n            cold_targets = [\n                symbol\n                for symbol in targets\n                if len(self._indicator_engine.get_history(symbol) or [])\n                < self._required_candles\n            ]\n            if not cold_targets:\n                self._logger.info(\n                    "⏭️ Skipping historical backfill (indicators fully warmed up)"\n                )\n                return\n\n            self._logger.warning(\n                "⚠️ StrategyRunner history short! Triggering targeted fallback backfill..."\n            )\n\n            for symbol in cold_targets:\n                try:\n                    target = self._required_bars_for_symbol(symbol)\n                    rows = self._get_mdm_bars(symbol, target)\n                    if rows:\n                        runner_count = self.reseed_history_from_bars(\n                            symbol,\n                            rows,\n                            source="runner_fallback_backfill",\n                            min_bars=target,\n                        )\n                        total_bars += runner_count\n                        if runner_count >= target:\n                            self._set_symbol_hydration_state(\n                                symbol, SymbolState.READY\n                            )\n                        else:\n                            self._set_symbol_hydration_state(\n                                symbol, SymbolState.HYDRATING\n                            )\n                            self._request_mdm_hydration(symbol, target)\n                    else:\n                        self._set_symbol_hydration_state(\n                            symbol, SymbolState.HYDRATING\n                        )\n                        self._request_mdm_hydration(symbol, target)\n\n                except Exception as exc:\n                    self._set_symbol_hydration_state(symbol, SymbolState.HYDRATING)\n                    self._logger.error(\n                        "❌ Fallback fetch failed for %s: %s", symbol, exc\n                    )\n\n        except Exception as exc:\n            self._logger.error(\n                "❌ History backfill crashed: %s", exc, exc_info=True\n            )\n\n        if total_bars > 0:\n            self._logger.info(\n                "✅ Emergency Backfill complete. Reseeded %d bars.", total_bars\n            )\n\n''',
    )

    replace_once(
        MDM,
        "            threshold_ms = 120000\n",
        "            threshold_ms = max(1, int(self._tick_stale_threshold_ms))\n",
    )
    replace_once(
        MDM,
        '''        spot_ready = True\n        if spot:\n            spot_bar_ready = bars.get(spot, 0) >= min_bars\n            try:\n                spot_ready = bool(\n                    self._is_symbol_fresh(spot, self._tick_stale_threshold_ms)\n                )\n            except Exception as exc:\n                self._logger.error("Failure in _readiness_state: %s", exc, exc_info=exc)\n                spot_ready = False\n            if not spot_ready and spot_bar_ready:\n                spot_ready = True\n''',
        '''        spot_ready = True\n        if spot:\n            try:\n                spot_ready = bool(\n                    self._is_symbol_fresh(spot, self._tick_stale_threshold_ms)\n                )\n            except Exception as exc:\n                self._logger.error("Failure in _readiness_state: %s", exc, exc_info=exc)\n                spot_ready = False\n            if not spot_ready:\n                missing_hard.append("fresh_spot_tick_missing")\n''',
    )
    print("NARROW_FIX_APPLIED", flush=True)


def commit_clean_diff() -> None:
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    run(
        "git",
        "rm",
        "-f",
        str(WORKFLOW.relative_to(ROOT)),
        str(SCRIPT.relative_to(ROOT)),
    )
    run(
        "git",
        "add",
        str(RUNNER.relative_to(ROOT)),
        str(MDM.relative_to(ROOT)),
        str(TEST.relative_to(ROOT)),
    )
    staged = run("git", "diff", "--cached", "--name-only")
    expected = {
        str(RUNNER.relative_to(ROOT)),
        str(MDM.relative_to(ROOT)),
        str(TEST.relative_to(ROOT)),
        str(WORKFLOW.relative_to(ROOT)),
        str(SCRIPT.relative_to(ROOT)),
    }
    actual = {line.strip() for line in staged.stdout.splitlines() if line.strip()}
    if actual != expected:
        raise RuntimeError(f"unexpected staged files: {sorted(actual)}")
    run(
        "git",
        "commit",
        "-m",
        "fix(runtime): preserve readiness and history integrity",
    )
    run("git", "push", "origin", "HEAD")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("red", "apply", "commit"))
    phase = parser.parse_args().phase
    if phase == "red":
        prove_red()
    elif phase == "apply":
        apply_fix()
    else:
        commit_clean_diff()


if __name__ == "__main__":
    main()
