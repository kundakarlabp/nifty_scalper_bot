from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ops.monitoring.nightly_replay import run_replay
import pytest


def _write_sample(path: Path, *, pnl: float, latency: float, mismatches: int) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pnl": pnl,
        "latency_p95": latency,
        "mismatches": mismatches,
        "errors": [],
    }
    path.write_text(json.dumps(payload))


def test_run_replay_generates_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    _write_sample(data_dir / "day1.json", pnl=125.5, latency=0.42, mismatches=0)
    _write_sample(data_dir / "day2.json", pnl=-10.0, latency=0.6, mismatches=2)

    artefact = run_replay(data_dir, report_dir)

    assert artefact.exists()
    report = json.loads(artefact.read_text())
    assert report["inputs_processed"] == 2
    assert report["pnl_total"] == pytest.approx(115.5, rel=1e-6)
    assert report["mismatches"] == 2
    assert report["latency_p95"] is not None


def test_run_replay_handles_missing_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()

    artefact = run_replay(data_dir, report_dir)
    assert artefact.exists()
    report = json.loads(artefact.read_text())
    assert report["inputs_processed"] == 0
    assert report["pnl_total"] == 0.0
    assert report["mismatches"] == 0
