#!/usr/bin/env python3
"""Unseal owner's Breeze research ZIPs into private verified offline windows.

Ownership: offline, human-invoked research transport only.
Safety: private key never read into Python or logged; no network, broker or orders.
Encrypted originals stay unchanged. Decrypted data must remain on private storage.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

from scripts.prepare_breeze_research_dataset import verify_window

SHA256 = re.compile(r"^([a-fA-F0-9]{64})  history.cms\s*$")
LIMIT_BYTES = 4 * 1024**3


def _safe_unseal(
    archive: Path, private_key: Path, recipient: Path, destination: Path
) -> None:
    if not archive.is_file() or destination.exists():
        raise ValueError("source_missing_or_window_already_unsealed")
    with tempfile.TemporaryDirectory(prefix="breeze-unseal-") as scratch:
        temporary = Path(scratch)
        with zipfile.ZipFile(archive) as src:
            members = {
                item.filename: item for item in src.infolist() if not item.is_dir()
            }
            if set(members) != {"history.cms", "history.cms.sha256"}:
                raise ValueError("encrypted_archive_members_invalid")
            if members["history.cms"].file_size > LIMIT_BYTES:
                raise ValueError("encrypted_archive_member_oversized")
            expected = SHA256.fullmatch(
                src.read("history.cms.sha256").decode("ascii").strip()
            )
            if expected is None:
                raise ValueError("encrypted_archive_checksum_invalid")
            cms = temporary / "history.cms"
            with src.open("history.cms") as reader, cms.open("wb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(chunk)
            if (
                hashlib.sha256(cms.read_bytes()).hexdigest()
                != expected.group(1).lower()
            ):
                raise ValueError("encrypted_archive_checksum_mismatch")
        decrypted_tar = temporary / "window.tar.gz"
        with subprocess.Popen(
            [
                "openssl",
                "cms",
                "-decrypt",
                "-binary",
                "-inform",
                "DER",
                "-in",
                str(cms),
                "-inkey",
                str(private_key),
                "-recip",
                str(recipient),
                "-out",
                str(decrypted_tar),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            if process.wait() != 0:
                raise ValueError("private_archive_decryption_failed")
        prepared = temporary / "extracted"
        prepared.mkdir(mode=0o700)
        with tarfile.open(decrypted_tar, "r:gz") as content:
            names: set[str] = set()
            total = 0
            for item in content.getmembers():
                normalized = item.name.removeprefix("./")
                path = Path(normalized)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (item.isfile() or item.isdir())
                    or normalized in names
                ):
                    raise ValueError("archive_tar_member_unsafe")
                names.add(normalized)
                if item.isfile():
                    total += item.size
                if total > LIMIT_BYTES:
                    raise ValueError("archive_tar_uncompressed_oversized")
            # Python 3.12+ explicit data filter; older versions validated above.
            for item in content.getmembers():
                normalized = item.name.removeprefix("./")
                if not normalized or normalized == ".":
                    continue
                target = prepared / normalized
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    stream = content.extractfile(item)
                    if stream is None:
                        raise ValueError("archive_tar_member_unreadable")
                    with stream, target.open("xb") as out:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            out.write(chunk)
                    os.chmod(target, 0o600)
        verify_window(prepared)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(prepared, destination)


def unseal(
    archives_dir: Path,
    private_key: Path,
    recipient: Path,
    output: Path,
    *,
    expected_windows: int = 25,
) -> dict:
    if (
        not archives_dir.is_dir()
        or not private_key.is_file()
        or not recipient.is_file()
        or not 1 <= expected_windows <= 100
    ):
        raise ValueError("unseal_inputs_missing")
    archives = sorted(archives_dir.glob("Breeze_Window_*.zip"))
    if len(archives) != expected_windows:
        raise ValueError("encrypted_window_count_mismatch")
    if output.exists() and any(output.iterdir()):
        raise ValueError("unseal_output_not_empty")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    for archive in archives:
        _safe_unseal(archive, private_key, recipient, output / archive.stem)
    return {
        "unsealed_verified_windows": len(archives),
        "all_output_private": True,
        "encrypted_inputs_modified": False,
        "execution_grade_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encrypted-windows", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--recipient-cert", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-windows", type=int, default=25)
    args = parser.parse_args()
    os.umask(0o077)
    result = unseal(
        args.encrypted_windows,
        args.private_key,
        args.recipient_cert,
        args.output,
        expected_windows=args.expected_windows,
    )
    print(
        f"Verified {result['unsealed_verified_windows']} owner-only windows; "
        "source ZIPs unchanged. Research only."
    )


if __name__ == "__main__":
    main()
