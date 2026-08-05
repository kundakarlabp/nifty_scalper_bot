from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDM = ROOT / "src/nifty_scalper_bot/data/market_data_manager.py"
RUNNER = ROOT / "src/nifty_scalper_bot/strategies/runner.py"
TEST = ROOT / "tests/runtime/test_readiness_history_integrity.py"
WORKFLOW = ROOT / ".github/workflows/one-shot-runtime-readiness-history-tdd.yml"
SCRIPT = Path(__file__).resolve()


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


mdm = MDM.read_text()
old_spot = '''        spot_ready = True
        if spot:
            spot_bar_ready = bars.get(spot, 0) >= min_bars
            try:
                spot_ready = bool(
                    self._is_symbol_fresh(spot, self._tick_stale_threshold_ms)
                )
            except Exception as exc:
                self._logger.error("Failure in _readiness_state: %s", exc, exc_info=exc)
                spot_ready = False
            if not spot_ready and spot_bar_ready:
                spot_ready = True
'''
new_spot = '''        spot_ready = True
        if spot:
            try:
                spot_ready = bool(
                    self._is_symbol_fresh(spot, self._tick_stale_threshold_ms)
                )
            except Exception as exc:
                self._logger.error("Failure in _readiness_state: %s", exc, exc_info=exc)
                spot_ready = False
'''
if old_spot not in mdm:
    raise SystemExit("spot readiness block not found")
MDM.write_text(mdm.replace(old_spot, new_spot, 1))

runner = RUNNER.read_text()
old_activation = '''                self._active_symbols.add(normalized_symbol)
                self._tracked_symbols.add(normalized_symbol)
                self._data_phase.setdefault(normalized_symbol, "HYDRATION")
'''
new_activation = '''                self._data_phase.setdefault(normalized_symbol, "HYDRATION")
'''
if old_activation not in runner:
    raise SystemExit("reseed activation block not found")
runner = runner.replace(old_activation, new_activation, 1)

old_backfill = '''                    if rows:
                        for bar_data in rows:
                            self.ingest_historical_bar(bar_data)
                            total_bars += 1
                        if len(rows) >= target:
                            self._set_symbol_hydration_state(symbol, SymbolState.READY)
                        else:
                            self._set_symbol_hydration_state(
                                symbol, SymbolState.HYDRATING
                            )
                            self._request_mdm_hydration(symbol, target)
'''
new_backfill = '''                    if rows:
                        reseeded = self.reseed_history_from_bars(
                            symbol,
                            rows,
                            source="runner_fallback",
                            min_bars=target,
                        )
                        total_bars += int(reseeded or 0)
                        if len(rows) >= target:
                            self._set_symbol_hydration_state(symbol, SymbolState.READY)
                        else:
                            self._set_symbol_hydration_state(
                                symbol, SymbolState.HYDRATING
                            )
                            self._request_mdm_hydration(symbol, target)
'''
if old_backfill not in runner:
    raise SystemExit("fallback append block not found")
RUNNER.write_text(runner.replace(old_backfill, new_backfill, 1))

run("python", "-m", "pytest", str(TEST), "-q")
run("python", "-m", "compileall", "-q", str(MDM), str(RUNNER))
run("python", "-m", "ruff", "check", str(MDM.relative_to(ROOT)), str(RUNNER.relative_to(ROOT)), str(TEST.relative_to(ROOT)))
run("python", "-m", "black", "--check", str(MDM.relative_to(ROOT)), str(RUNNER.relative_to(ROOT)), str(TEST.relative_to(ROOT)))

run("git", "rm", "-f", str(WORKFLOW.relative_to(ROOT)), str(SCRIPT.relative_to(ROOT)))
run("git", "add", str(MDM.relative_to(ROOT)), str(RUNNER.relative_to(ROOT)), str(TEST.relative_to(ROOT)))
run("git", "commit", "-m", "fix(runtime): preserve readiness and history ownership")
run("git", "push", "origin", "HEAD")
