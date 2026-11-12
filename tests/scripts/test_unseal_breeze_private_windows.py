"""Offline CMS transport regression: no broker secrets, no network."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile

import pytest

from scripts.prepare_breeze_research_dataset import build, digest_json
from scripts.unseal_breeze_private_windows import unseal


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL unavailable")
def test_encrypted_window_round_trip_to_private_research_sqlite(tmp_path):
    key = tmp_path / "private.pem"
    cert = tmp_path / "recipient.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=offline-research-fixture",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    payload = {
        "schema_version": 1, "day": "2025-02-03",
        "expiry": "2025-02-27", "strike": 22500, "side": "PE",
        "source": "icici_breeze_historical_v2",
        "product_type": "options",
        "normalized_timestamp_convention": "bar_start",
        "promotion_eligible": False, "candles": [], "provider_rows": [],
        "candles_sha256": digest_json([]),
        "provider_rows_sha256": digest_json([]),
    }
    # Correct canonical Breeze key is provider_sha256.
    payload["provider_sha256"] = payload.pop("provider_rows_sha256")
    body = json.dumps(payload).encode()
    member = "breeze_history/2025-02-03/NIFTY_PE.json"
    checksum = f"{hashlib.sha256(body).hexdigest()}  {member}\n".encode()
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        for name, data in ((member, body), ("SHA256SUMS", checksum)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    raw_tar = tmp_path / "raw.tar.gz"
    raw_tar.write_bytes(tar_bytes.getvalue())
    cms = tmp_path / "history.cms"
    subprocess.run(
        [
            "openssl", "cms", "-encrypt", "-binary", "-aes-256-gcm",
            "-outform", "DER", "-in", str(raw_tar),
            "-out", str(cms), str(cert),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    folder = tmp_path / "zips"
    folder.mkdir()
    zip_path = folder / "Breeze_Window_fixture.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(cms, "history.cms")
        archive.writestr(
            "history.cms.sha256",
            f"{hashlib.sha256(cms.read_bytes()).hexdigest()}  history.cms\n",
        )
    dest = tmp_path / "private" / "windows"
    result = unseal(folder, key, cert, dest, expected_windows=1)
    assert result["unsealed_verified_windows"] == 1
    assert zip_path.exists()
    report = build(
        dest, tmp_path / "private" / "candles.sqlite",
        tmp_path / "private" / "summary.json", expected_windows=1
    )
    assert report["verified_windows"] == 1
    assert report["series"] == 1
    assert report["bars_1m"] == 0
    assert report["execution_grade_ready"] is False


def test_missing_private_key_fails_closed(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    with pytest.raises(ValueError, match="unseal_inputs_missing"):
        unseal(
            folder, tmp_path / "no-key.pem", tmp_path / "no-cert.pem",
            tmp_path / "private", expected_windows=1
        )
