"""Offline research pipeline tests for public-key matching and safe unseal.

Ownership: Research-only synthetic fixtures; no real owner keys or broker I/O.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.finish_breeze_private_replay import (
    execute,
    matching_key,
    verified_transports,
)
from scripts.prepare_breeze_research_dataset import digest_json


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    key = tmp_path / "matching-key.pem"
    cert = tmp_path / "recipient.pem"
    subprocess.run(
        [
            "openssl", "req", "-newkey", "rsa:2048", "-nodes",
            "-x509", "-days", "1", "-subj", "/CN=research-only-fixture",
            "-keyout", str(key), "-out", str(cert),
        ],
        check=True,
        capture_output=True,
    )
    bars = [
        [
            datetime(
                2025, 2, 3, 9, 15,
                tzinfo=timezone(timedelta(hours=5, minutes=30)),
            ).isoformat(),
            100.0, 101.0, 99.0, 100.5, 100, 1000,
        ]
    ]
    raw = []
    payload = {
        "schema_version": 1,
        "day": date(2025, 2, 3).isoformat(),
        "expiry": "2025-02-27",
        "strike": 22500,
        "side": "CE",
        "product_type": "options",
        "source": "icici_breeze_historical_v2",
        "normalized_timestamp_convention": "bar_start",
        "provider_rows": raw,
        "provider_sha256": digest_json(raw),
        "candles": bars,
        "candles_sha256": digest_json(bars),
        "promotion_eligible": False,
    }
    body = json.dumps(payload).encode()
    member = "breeze_history/2025-02-03/CE.json"
    checksums = f"{hashlib.sha256(body).hexdigest()}  {member}\n".encode()
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w:gz") as packed:
        for name, contents in (
            (member, body),
            ("SHA256SUMS", checksums),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(contents)
            packed.addfile(entry, io.BytesIO(contents))
    source = tmp_path / "fixture.tar.gz"
    source.write_bytes(tar.getvalue())
    cms = tmp_path / "history.cms"
    subprocess.run(
        [
            "openssl", "cms", "-encrypt", "-binary", "-aes-256-gcm",
            "-outform", "DER", "-in", str(source),
            "-out", str(cms), str(cert),
        ],
        check=True,
        capture_output=True,
    )
    folder = tmp_path / "sealed"
    folder.mkdir()
    with zipfile.ZipFile(folder / "Breeze_Window_fixture.zip", "w") as zipped:
        zipped.write(cms, "history.cms")
        zipped.writestr(
            "history.cms.sha256",
            f"{hashlib.sha256(cms.read_bytes()).hexdigest()}  history.cms\n",
        )
    return key, cert, folder, cms


def test_match_and_convert_private_research_archive(tmp_path: Path) -> None:
    key, cert, folder, _ = _fixture(tmp_path)
    assert matching_key(cert, search_directories=(tmp_path,)) == key
    assert verified_transports(folder, expected_windows=1)["verified_archives"] == 1
    output = tmp_path / "derived"
    result = execute(
        archives=folder, certificate=cert, output_root=output,
        search_directories=(tmp_path,), expected_windows=1,
    )
    assert result["intraday_sqlite_built"]
    assert result["unsealed_windows"] == 1
    assert result["source_reconstruction"]["bars_1m"] == 1
    assert result["source_reconstruction"]["execution_grade_ready"] is False
    assert result["private_key_material_exported"] is False
    assert (output / "pipeline_readiness.json").exists()
    assert key.exists()


def test_missing_key_does_not_create_output(tmp_path: Path) -> None:
    key, cert, folder, _ = _fixture(tmp_path)
    output = tmp_path / "derived"
    isolated = tmp_path / "empty"
    isolated.mkdir()
    result = execute(
        archives=folder, certificate=cert, output_root=output,
        search_directories=(isolated,), expected_windows=1,
    )
    assert key.exists()
    assert not output.exists()
    assert not result["matching_owner_key_found"]
    assert result["blocker"] == "original_cms_private_key_not_accessible_here"


def test_preflight_checks_cms_without_key(tmp_path: Path) -> None:
    _, cert, folder, _ = _fixture(tmp_path)
    result = execute(
        archives=folder, certificate=cert,
        output_root=None, preflight_only=True, expected_windows=1,
    )
    assert result["verified_archives"] == 1
    assert not result["key_checked_by_public_fingerprint"]


def test_wrong_direct_key_does_not_decrypt(tmp_path: Path) -> None:
    _, cert, folder, _ = _fixture(tmp_path)
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    wrong_key, _, _, _ = _fixture(wrong_dir)
    output = tmp_path / "derived"
    result = execute(
        archives=folder, certificate=cert, output_root=output,
        direct_key=wrong_key, expected_windows=1,
    )
    assert not result["matching_owner_key_found"]
    assert not output.exists()


def test_modified_cms_member_is_rejected_before_key_search(tmp_path: Path) -> None:
    _, _, folder, _ = _fixture(tmp_path)
    path = folder / "Breeze_Window_fixture.zip"
    with zipfile.ZipFile(path) as source:
        cms = source.read("history.cms")
        checksum = source.read("history.cms.sha256")
    with zipfile.ZipFile(path, "w") as destination:
        destination.writestr("history.cms", cms + b"tampering")
        destination.writestr("history.cms.sha256", checksum)
    with pytest.raises(ValueError, match="archive_transport_checksum_mismatch"):
        verified_transports(folder, expected_windows=1)
