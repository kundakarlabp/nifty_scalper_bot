#!/usr/bin/env python3
"""Unblock private Breeze intraday research with an owner-local CMS key.

Responsibilities: Verify sealed 25-window transports, locate a matching local
key by public-key fingerprint, decrypt and build a separate research-only DB.
Ownership: Offline private workstation. Never invoked by the live runner.
Safety: No key material, trading credentials, remote calls or source mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from scripts.prepare_breeze_research_dataset import build
from scripts.unseal_breeze_private_windows import unseal

ORIGINAL_WINDOW_BYTES = 107_821_151
ORIGINAL_MINUTE_OBSERVATIONS = 1_454_867
SKIP_DIRS = frozenset(
    (".git", ".cache", ".venv", "venv", "node_modules", "__pycache__", "site-packages")
)
KEY_EXTENSIONS = frozenset((".pem", ".key", ".p8"))
MAX_CANDIDATES = 5000


def _openssl(args: list[str], *, input_bytes: bytes | None = None) -> bytes | None:
    try:
        result = subprocess.run(
            ["openssl", *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _certificate_fingerprint(certificate: Path) -> bytes:
    pem = _openssl(["x509", "-pubkey", "-noout", "-in", str(certificate)])
    if pem is None:
        raise ValueError("recipient_certificate_unreadable")
    der = _openssl(["pkey", "-pubin", "-outform", "DER"], input_bytes=pem)
    if der is None:
        raise ValueError("recipient_public_key_unreadable")
    return hashlib.sha256(der).digest()


def matching_key(
    certificate: Path,
    *,
    direct_key: Path | None = None,
    search_directories: tuple[Path, ...] = (),
) -> Path | None:
    """Compare only derived public key hashes; never return private key contents."""
    wanted = _certificate_fingerprint(certificate)
    if direct_key is not None:
        candidates = [direct_key]
    else:
        candidates = []
        inspected = 0
        for root in search_directories:
            if not root.is_dir():
                continue
            for base, directories, filenames in os.walk(root, followlinks=False):
                directories[:] = sorted(d for d in directories if d not in SKIP_DIRS)
                for filename in sorted(filenames):
                    file = Path(base) / filename
                    if file.suffix.lower() not in KEY_EXTENSIONS or file.is_symlink():
                        continue
                    try:
                        if file.stat().st_size > 1024 * 1024:
                            continue
                    except OSError:
                        continue
                    inspected += 1
                    if inspected > MAX_CANDIDATES:
                        raise ValueError("key_candidate_limit_exceeded")
                    candidates.append(file)
    for file in candidates:
        if not file.is_file() or file.is_symlink():
            continue
        public = _openssl(
            [
                "pkey",
                "-pubout",
                "-outform",
                "DER",
                "-passin",
                "pass:",
                "-in",
                str(file),
            ]
        )
        if public is not None and hashlib.sha256(public).digest() == wanted:
            return file.resolve()
    return None


def verified_transports(
    directory: Path, *, expected_windows: int = 25
) -> dict[str, int]:
    """Verify transport member names and CMS SHA-256 before any key use."""
    if not directory.is_dir():
        raise ValueError("encrypted_archive_directory_missing")
    archives = sorted(directory.glob("Breeze_Window_*.zip"))
    if len(archives) != expected_windows:
        raise ValueError("original_archive_count_mismatch")
    total_bytes = 0
    for path in archives:
        if path.is_symlink():
            raise ValueError("archive_symlink_not_permitted")
        total_bytes += path.stat().st_size
        with zipfile.ZipFile(path) as source:
            if sorted(source.namelist()) != ["history.cms", "history.cms.sha256"]:
                raise ValueError("archive_members_invalid")
            digest_line = source.read("history.cms.sha256").decode("ascii").strip()
            checksum, sep, filename = digest_line.partition("  ")
            if (
                not sep
                or filename != "history.cms"
                or len(checksum) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in checksum)
            ):
                raise ValueError("archive_transport_checksum_invalid")
            running = hashlib.sha256()
            with source.open("history.cms") as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    running.update(chunk)
            if running.hexdigest() != checksum.lower():
                raise ValueError("archive_transport_checksum_mismatch")
    if expected_windows == 25 and total_bytes != ORIGINAL_WINDOW_BYTES:
        raise ValueError("original_archive_total_size_mismatch")
    return {"verified_archives": len(archives), "encrypted_bytes": total_bytes}


def execute(
    *,
    archives: Path | None,
    certificate: Path,
    output_root: Path | None,
    direct_key: Path | None = None,
    search_directories: tuple[Path, ...] = (),
    expected_windows: int = 25,
    preflight_only: bool = False,
    extracted_windows: Path | None = None,
    eod_db: Path | None = None,
    run_strategies: bool = False,
    permit_partial: bool = False,
) -> dict[str, Any]:
    """Never mutate source; accept either old CMS or verified plaintext windows."""
    if (archives is None) == (extracted_windows is None):
        raise ValueError("select_one_historical_source")
    if extracted_windows is None:
        observed = verified_transports(archives, expected_windows=expected_windows)
    else:
        if not extracted_windows.is_dir():
            raise ValueError("plaintext_research_source_missing")
        observed = {"verified_archives": 0, "encrypted_bytes": 0}
    report: dict[str, Any] = {
        **observed,
        "key_checked_by_public_fingerprint": False,
        "matching_owner_key_found": False,
        "private_key_material_exported": False,
        "research_only": True,
        "live_promotion_eligible": False,
        "source_archives_modified": False,
        "intraday_sqlite_built": False,
        "original_cms_key_not_required": extracted_windows is not None,
    }
    if preflight_only:
        return report
    if extracted_windows is None:
        key = matching_key(
            certificate, direct_key=direct_key, search_directories=search_directories
        )
        report["key_checked_by_public_fingerprint"] = True
        report["matching_owner_key_found"] = bool(key)
        if key is None:
            report["blocker"] = "original_cms_private_key_not_accessible_here"
            return report
    else:
        key = None
    if output_root is None:
        raise ValueError("private_output_root_required")
    output = output_root.resolve()
    source = (extracted_windows or archives).resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        or output.exists()
    ):
        raise ValueError("private_output_must_be_new_and_outside_source")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    if extracted_windows is None:
        unseal_result = unseal(
            archives,
            key,
            certificate,
            output / "windows",
            expected_windows=expected_windows,
        )
        source_windows = output / "windows"
    else:
        unseal_result = {"unsealed_verified_windows": 0}
        source_windows = extracted_windows
    converted = build(
        source_windows,
        output / "nifty_intraday_research.sqlite",
        output / "coverage_summary.json",
        expected_windows=expected_windows,
    )
    admitted_plus_excluded = (
        converted["bars_1m"]
        + converted["quarantined_source_minutes"]
        + converted["special_boundary_exclusions"]
    )
    report.update(
        {
            "intraday_sqlite_built": True,
            "unsealed_windows": unseal_result["unsealed_verified_windows"],
            "source_reconstruction": converted,
            "minute_reconciliation": {
                "original_audit_minutes": ORIGINAL_MINUTE_OBSERVATIONS,
                "admitted_plus_explicit_exclusions": admitted_plus_excluded,
                "exact_match_to_prior_audit": (
                    expected_windows == 25
                    and admitted_plus_excluded == ORIGINAL_MINUTE_OBSERVATIONS
                    and converted["identical_duplicates"] == 0
                ),
            },
            "execution_grade_ready": False,
        }
    )
    if eod_db is not None:
        if expected_windows == 25 and not report["minute_reconciliation"][
            "exact_match_to_prior_audit"
        ]:
            report["offline_research_blocker"] = "source_minute_audit_not_reconciled"
        else:
            from scripts.stage_breeze_sqlite_research import run

            stage = run(
                output / "nifty_intraday_research.sqlite",
                eod_db,
                output / "research_sessions",
                summary_path=output / "coverage_summary.json",
                expected_windows=expected_windows,
                permit_partial=permit_partial,
                execute=run_strategies,
            )
            report["offline_research"] = stage
    (output / "pipeline_readiness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    os.chmod(output / "pipeline_readiness.json", 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--archives", type=Path)
    source_group.add_argument("--extracted-windows", type=Path)
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path("deploy/breeze_transport_recipient.pem"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--key-search-dir", action="append", type=Path, default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--eod-db", type=Path)
    parser.add_argument("--run-strategies", action="store_true")
    parser.add_argument("--permit-partial", action="store_true")
    args = parser.parse_args()
    if args.run_strategies and args.eod_db is None:
        parser.error("--run-strategies requires --eod-db")
    roots = tuple(args.key_search_dir) or (
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / ".config" / "niftyscalper",
    )
    result = execute(
        archives=args.archives,
        certificate=args.certificate,
        output_root=args.output_root,
        direct_key=args.key,
        search_directories=roots,
        preflight_only=args.preflight_only,
        extracted_windows=args.extracted_windows,
        eod_db=args.eod_db,
        run_strategies=args.run_strategies,
        permit_partial=args.permit_partial,
    )
    # Print only bounded status fields: no secret, arbitrary path or raw bars.
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "verified_archives",
                    "encrypted_bytes",
                    "matching_owner_key_found",
                    "intraday_sqlite_built",
                    "blocker",
                    "minute_reconciliation",
                    "execution_grade_ready",
                    "original_cms_key_not_required",
                    "offline_research",
                    "offline_research_blocker",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if args.preflight_only or result["intraday_sqlite_built"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
