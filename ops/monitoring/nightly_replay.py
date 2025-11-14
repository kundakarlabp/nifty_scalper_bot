"""Nightly replay automation to validate live trading against backtests."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ReplaySummary:
    """Container describing the outcome of a nightly replay run."""

    generated_at: datetime
    inputs_processed: int
    pnl_total: float
    latency_p95: float | None
    mismatches: int
    errors: list[str]

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable representation of the replay summary.

        Args:
            None.

        Returns:
            dict[str, object]: Canonical JSON-serialisable mapping of the summary.

        Raises:
            None.
        """

        return {
            "generated_at": self.generated_at.replace(tzinfo=timezone.utc).isoformat(),
            "inputs_processed": self.inputs_processed,
            "pnl_total": round(self.pnl_total, 2),
            "latency_p95": self.latency_p95,
            "mismatches": self.mismatches,
            "errors": list(self.errors),
        }


def _iter_input_files(source: Path) -> Iterable[Path]:
    """Yield candidate replay input files from *source* directory.

    Args:
        source: Directory path expected to contain nightly replay inputs.

    Returns:
        Iterable[Path]: Iterable of JSON files sorted lexicographically.

    Raises:
        None.
    """

    if not source.exists():
        return []
    return sorted(path for path in source.glob("*.json") if path.is_file())


def run_replay(data_dir: Path, output_dir: Path) -> Path:
    """Execute nightly replay pipeline and return generated report path.

    Args:
        data_dir: Directory containing nightly input snapshots.
        output_dir: Destination directory for generated reports.

    Returns:
        Path to the summary JSON artefact.

    Raises:
        None.
    """

    LOGGER.debug("Entered run_replay", extra={"event": "nightly_replay_start"})
    inputs = list(_iter_input_files(data_dir))
    pnl_total = 0.0
    latency_samples: list[float] = []
    mismatches = 0
    errors: list[str] = []
    for candidate in inputs:
        try:
            payload = json.loads(candidate.read_text())
        except Exception as exc:  # noqa: BLE001 - defensive parsing
            LOGGER.error(
                "Failed to load replay input: %s",
                exc,
                extra={"file": str(candidate)},
            )
            errors.append(f"load_error:{candidate.name}")
            continue
        pnl_total += float(payload.get("pnl", 0.0) or 0.0)
        latency = payload.get("latency_p95")
        if isinstance(latency, (int, float)):
            latency_samples.append(float(latency))
        mismatches += int(payload.get("mismatches", 0) or 0)
        for marker in payload.get("errors", []) or []:
            errors.append(str(marker))
    latency_p95 = None
    if latency_samples:
        sorted_samples = sorted(latency_samples)
        index = max(0, int(len(sorted_samples) * 0.95) - 1)
        latency_p95 = round(sorted_samples[index], 4)
    summary = ReplaySummary(
        generated_at=datetime.now(timezone.utc),
        inputs_processed=len(inputs),
        pnl_total=pnl_total,
        latency_p95=latency_p95,
        mismatches=mismatches,
        errors=errors[-10:],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artefact = output_dir / "nightly_report.json"
    artefact.write_text(json.dumps(summary.as_dict(), indent=2))
    try:
        METRICS.increment_retry_event(
            label="nightly_replay", stage="run", outcome="success"
        )
    except Exception:  # noqa: BLE001 - optional metrics backend
        LOGGER.debug("Metrics backend unavailable for nightly replay")
    LOGGER.info(
        "Nightly replay completed",
        extra={
            "event": "nightly_replay_complete",
            "summary": summary.as_dict(),
        },
    )
    return artefact


def main() -> int:
    """Entry point for CLI usage.

    Args:
        None.

    Returns:
        int: Process exit status (``0`` on success).

    Raises:
        None.
    """

    base = Path.cwd()
    data_dir = base / "data" / "nightly" / "inputs"
    output_dir = base / "data" / "nightly" / "reports"
    artefact = run_replay(data_dir, output_dir)
    print(artefact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    sys.exit(main())
