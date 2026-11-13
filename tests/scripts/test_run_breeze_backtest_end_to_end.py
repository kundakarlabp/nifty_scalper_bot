"""Private-key workflow preserves confidentiality and fails before importing data."""

import subprocess

import pytest
from scripts.run_breeze_backtest_end_to_end import (
    execute_workflow,
    locate_matching_key,
)


def _temporary_cert(tmp_path):
    key = tmp_path / "test-only-recipient.key"
    cert = tmp_path / "test-only-recipient.pem"
    subprocess.run(
        [
            "openssl", "req", "-newkey", "rsa:2048", "-nodes", "-x509",
            "-days", "1", "-subj", "/CN=synthetic-fixture-only",
            "-keyout", str(key), "-out", str(cert),
        ],
        capture_output=True, check=True,
    )
    return key, cert


def test_public_certificate_fingerprint_locates_only_matching_key(tmp_path):
    key, cert = _temporary_cert(tmp_path)
    assert locate_matching_key(cert, [tmp_path]) == key.resolve()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert locate_matching_key(cert, [empty]) is None


def test_missing_original_key_rejects_without_creating_output(tmp_path):
    _, cert = _temporary_cert(tmp_path)
    original = tmp_path / "encrypted"
    original.mkdir()
    eod = tmp_path / "eod.sqlite"
    eod.touch()
    out = tmp_path / "research"
    separate = tmp_path / "no_private_keys"
    separate.mkdir()
    with pytest.raises(ValueError, match="original_matching_cms_key_unavailable"):
        execute_workflow(
            archive_dir=original,
            extracted_windows=None,
            recipient_cert=cert,
            eod_db=eod,
            output_root=out,
            search_roots=[separate],
        )
    assert not out.exists()


def test_live_execution_mode_rejected_even_before_key_search(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="live_execution_mode_not_allowed"):
        execute_workflow(
            archive_dir=None,
            extracted_windows=None,
            recipient_cert=tmp_path / "none",
            eod_db=tmp_path / "none",
            output_root=tmp_path / "output",
            search_roots=[],
        )