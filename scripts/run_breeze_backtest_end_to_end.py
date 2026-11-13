"""One-command owner-local import of Breeze data into offline strategy research.

Responsibilities: Locate an existing matching CMS key, verify/unseal existing
archives, build research SQLite and run the canonical strategy-research adapter.
Ownership: Human-invoked private workstation operation, not production runner.
Safety: Never exports private-key material, performs broker I/O, uses LIVE mode,
or weakens execution-grade safety checks; never changes original archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "site-packages", ".cache"}
KEY_SUFFIXES = {".pem", ".key", ".p8"}


def _public_digest(argv: list[str], *, input_data: bytes | None = None) -> bytes | None:
    try:
        result = subprocess.run(
            argv, input=input_data, capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return hashlib.sha256(result.stdout).digest()


def locate_matching_key(cert: Path, roots: list[Path]) -> Path | None:
    """Compare PUBLIC fingerprints, never display or copy the private key."""
    try:
        pub = subprocess.run(
            ["openssl", "x509", "-in", str(cert), "-pubkey", "-noout"],
            capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("recipient_public_certificate_unavailable") from None
    if pub.returncode:
        raise ValueError("recipient_public_certificate_unavailable")
    wanted = _public_digest(
        ["openssl", "pkey", "-pubin", "-outform", "DER"],
        input_data=pub.stdout,
    )
    if wanted is None:
        raise ValueError("recipient_certificate_public_key_unreadable")
    checked = 0
    for root in roots:
        if not root.is_dir():
            continue
        for parent, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            for filename in files:
                item = Path(parent) / filename
                if item.is_symlink() or item.suffix.lower() not in KEY_SUFFIXES:
                    continue
                if item.stat().st_size > 1024 * 1024:
                    continue
                checked += 1
                if checked > 5000:
                    raise ValueError("private_key_candidate_limit_exceeded")
                got = _public_digest(
                    [
                        "openssl", "pkey", "-in", str(item), "-pubout",
                        "-outform", "DER", "-passin", "pass:",
                    ]
                )
                if got == wanted:
                    return item.resolve()
    return None


def execute_workflow(
    *,
    archive_dir: Path | None,
    extracted_windows: Path | None,
    recipient_cert: Path,
    eod_db: Path,
    output_root: Path,
    search_roots: list[Path],
    direct_key: Path | None = None,
    expected_windows: int = 25,
    stage_only: bool = False,
) -> dict[str, Any]:
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("live_execution_mode_not_allowed")
    if os.getenv("ENABLE_LIVE", "false").lower() == "true":
        raise ValueError("live_execution_mode_not_allowed")
    if output_root.exists():
        raise ValueError("research_output_already_exists")
    if not eod_db.is_file() or not 1 <= expected_windows <= 25:
        raise ValueError("research_sources_missing")
    if extracted_windows is not None:
        if not extracted_windows.is_dir():
            raise ValueError("extracted_research_windows_unavailable")
        source = extracted_windows
    else:
        if archive_dir is None or not archive_dir.is_dir():
            raise ValueError("original_encrypted_archive_directory_missing")
        key = direct_key or locate_matching_key(recipient_cert, search_roots)
        if key is None or not key.is_file():
            raise ValueError("original_matching_cms_key_unavailable")
        source = output_root / "extracted"
    from scripts.prepare_breeze_research_dataset import build
    from scripts.stage_breeze_sqlite_research import run
    from scripts.unseal_breeze_private_windows import unseal

    output_root.mkdir(mode=0o700, parents=True)
    if extracted_windows is None:
        unseal(archive_dir, key, recipient_cert, source, expected_windows=expected_windows)
    research_db = output_root / "nifty_1m.sqlite"
    summary_path = output_root / "coverage_summary.json"
    verified = build(
        source, research_db, summary_path, expected_windows=expected_windows
    )
    stage = run(
        research_db, eod_db, output_root / "research",
        summary_path=summary_path, expected_windows=expected_windows,
        permit_partial=True, execute=not stage_only,
    )
    report = {
        "verified_windows": verified["verified_windows"],
        "imported_minute_bars": verified["bars_1m"],
        "special_session_exclusions": verified["special_boundary_exclusions"],
        "quarantined_source_minutes": verified["quarantined_source_minutes"],
        "strategy_stage": stage,
        "actual_intraday_backtests_completed": stage["research_executions_completed"],
        "data_and_runner_status": (
            "research_replay_completed"
            if stage["research_executions_completed"] > 0
            else "staged_only_or_no_eligible_sessions"
        ),
        "original_archive_modified": False,
        "private_key_exported": False,
        "historical_execution_quotes_available": False,
        "execution_grade_ready": False,
        "live_promotion_eligible": False,
    }
    (output_root / "final_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (output_root / "final_summary.json").chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encrypted-archives", type=Path)
    group.add_argument("--extracted-windows", type=Path)
    parser.add_argument("--recipient-cert", type=Path, required=True)
    parser.add_argument("--eod-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, action="append", default=[])
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    result = execute_workflow(
        archive_dir=args.encrypted_archives,
        extracted_windows=args.extracted_windows,
        recipient_cert=args.recipient_cert,
        eod_db=args.eod_db,
        output_root=args.output_root,
        search_roots=args.search_dir or [Path.home()],
        direct_key=args.key_file,
        stage_only=args.stage_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()