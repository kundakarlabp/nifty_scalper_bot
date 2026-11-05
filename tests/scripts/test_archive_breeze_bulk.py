"""Private acquisition must preserve identities, budgets, gaps and integrity."""

import csv
import datetime as dt
import gzip
import io
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from scripts import archive_breeze_bulk as bulk
from scripts.archive_breeze_history import collect_contract

DAY = dt.date(2025, 2, 3)
EXPIRY = dt.date(2025, 2, 6)


def zip_catalog(wrong_date=False):
    stream = io.StringIO()
    fields = ["BizDt", "TckrSymb", "FinInstrmTp", "XpryDt", "StrkPric", "OptnTp"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    base = {"BizDt": "2025-02-04" if wrong_date else str(DAY), "TckrSymb": "NIFTY"}
    writer.writerow({**base, "FinInstrmTp": "IDF", "XpryDt": "2025-02-27"})
    for expiry in ("2025-02-06", "2025-02-13", "2025-02-20"):
        for strike in (23200, 23250):
            for side in ("CE", "PE"):
                writer.writerow(
                    {
                        **base,
                        "FinInstrmTp": "IDO",
                        "XpryDt": expiry,
                        "StrkPric": strike,
                        "OptnTp": side,
                    }
                )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.csv", stream.getvalue())
    return payload.getvalue()


class Client:
    def __init__(self):
        self.calls = []

    def get_historical_data_v2(self, **request):
        self.calls.append(request)
        row = {
            "datetime": "2025-02-03 09:15:00",
            "stock_code": "NIFTY",
            "exchange_code": request["exchange_code"],
            "open": 23210,
            "high": 23220,
            "low": 23200,
            "close": 23210,
            "volume": 75,
            "open_interest": 750,
        }
        if request["product_type"] != "cash":
            expiry = dt.date.fromisoformat(request["expiry_date"][:10])
            row.update(
                product_type=request["product_type"].title(),
                expiry_date=expiry.strftime("%d-%b-%Y").upper(),
            )
        if request["product_type"] == "options":
            row.update(
                right=request["right"].title(),
                strike_price=float(request["strike_price"]),
            )
        return {"Status": 200, "Error": None, "Success": [row]}


def test_cash_and_future_use_same_sealed_owner_and_cache(tmp_path):
    client = Client()
    for kind, expiry, side in (
        ("cash", None, "INDEX"),
        ("futures", dt.date(2025, 2, 27), "FUT"),
    ):
        params = dict(
            day=DAY,
            expiry=expiry,
            strike=None,
            side=side,
            product_type=kind,
            outdir=tmp_path,
            timestamp_convention="bar_start",
        )
        result = collect_contract(client, **params)
        assert result["bars"] == 1
        assert result["missing_minutes"] == 374
        data = json.loads((tmp_path / result["path"]).read_text())
        assert data["product_type"] == kind
        assert data["source_timezone_verified"] is False
        assert data["promotion_eligible"] is False
        assert collect_contract(client, **params)["requests_made"] == 0
    assert len(client.calls) == 2


def test_future_wrong_expiry_and_invalid_product_fail(tmp_path):
    class Wrong(Client):
        def get_historical_data_v2(self, **request):
            data = super().get_historical_data_v2(**request)
            data["Success"][0]["expiry_date"] = "27-MAR-2025"
            return data

    params = dict(
        day=DAY,
        expiry=dt.date(2025, 2, 27),
        strike=None,
        side="FUT",
        product_type="futures",
        outdir=tmp_path,
        timestamp_convention="bar_start",
    )
    with pytest.raises(ValueError, match="identity_mismatch"):
        collect_contract(Wrong(), **params)
    with pytest.raises(ValueError, match="contract_or_date_invalid"):
        collect_contract(Client(), **{**params, "product_type": "cash"})


def test_catalog_preserves_all_published_rows_but_limits_download_universe():
    catalog = bulk.parse_catalog(zip_catalog(), DAY)
    assert len(catalog["all_nifty_eod_rows"]) == 13
    assert len(catalog["option_identities"]) == 12
    assert catalog["available_not_before"] == "2025-02-04"
    assert catalog["ex_post_reference_only"] is True
    selected = bulk.select_options(catalog, 23208)
    assert len(selected) == 4
    assert {strike for _, strike, _ in selected} == {23200}
    assert {expiry for expiry, _, _ in selected} == {"2025-02-06", "2025-02-13"}
    with pytest.raises(ValueError, match="date_mismatch"):
        bulk.parse_catalog(zip_catalog(True), DAY)


def test_budget_stops_before_the_next_provider_request(monkeypatch):
    monkeypatch.setattr(bulk.time, "sleep", lambda _: None)
    client = Client()
    wrapper = bulk.ResearchClient(client, 1, time.monotonic() + 10)
    wrapper.get_historical_data_v2(product_type="cash", exchange_code="NSE")
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        wrapper.get_historical_data_v2(product_type="cash", exchange_code="NSE")
    assert len(client.calls) == 1


def test_checkpoint_and_hashes_survive_partial_day(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk, "fetch", lambda *_: zip_catalog())
    monkeypatch.setattr(bulk.time, "sleep", lambda _: None)
    report = bulk.acquire(
        bulk.ResearchClient(Client(), 2, time.monotonic() + 20),
        DAY,
        DAY + dt.timedelta(days=1),
        tmp_path,
        {bulk.catalog_path(DAY)},
    )
    assert report["requests_made"] == 2
    assert report["stopped_early"] is True
    assert report["collection_complete"] is False
    assert report["total_bars"] == 2
    assert (tmp_path / "SHA256SUMS").is_file()
    assert (tmp_path / "coverage.json").is_file()
    catalog = json.loads(
        gzip.decompress((tmp_path / "catalog" / f"{DAY}.json.gz").read_bytes())
    )
    assert len(catalog["all_nifty_eod_rows"]) == 13


def test_full_day_core_scope_does_not_claim_all_intraday_strikes(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk, "fetch", lambda *_: zip_catalog())
    monkeypatch.setattr(bulk.time, "sleep", lambda _: None)
    report = bulk.acquire(
        bulk.ResearchClient(Client(), 16, time.monotonic() + 20),
        DAY,
        DAY + dt.timedelta(days=1),
        tmp_path,
        {bulk.catalog_path(DAY)},
    )
    assert report["requests_made"] == 6
    assert report["plan_walk_completed"] is True
    assert report["all_historical_strikes_collected"] is False
    assert report["quotes_or_depth_reconstructed"] is False
    assert report["promotion_eligible"] is False


def test_unavailable_catalog_does_not_invent_futures_or_options(tmp_path, monkeypatch):
    def unavailable(*_):
        raise OSError("source unavailable")

    monkeypatch.setattr(bulk, "fetch", unavailable)
    report = bulk.acquire(
        bulk.ResearchClient(Client(), 16, time.monotonic() + 20),
        DAY,
        DAY + dt.timedelta(days=1),
        tmp_path,
        set(),
    )
    assert report["source_unavailable"] == [str(DAY)]
    assert report["collection_complete"] is False
    assert [c["kind"] for c in report["contracts"]] == ["cash"]


def test_ciphertext_authentication_rejects_tampering(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL not installed")

    def run(*args):
        return subprocess.run(
            ["openssl", *args],
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    assert (
        run(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            "key.pem",
            "-out",
            "cert.pem",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=fixture",
        ).returncode
        == 0
    )
    (tmp_path / "raw").write_bytes(b"fixture market rows")
    assert (
        run(
            "cms",
            "-encrypt",
            "-binary",
            "-aes-256-gcm",
            "-in",
            "raw",
            "-out",
            "transport.cms",
            "-outform",
            "DER",
            "cert.pem",
        ).returncode
        == 0
    )
    decrypt = (
        "cms",
        "-decrypt",
        "-binary",
        "-inform",
        "DER",
        "-in",
        "transport.cms",
        "-recip",
        "cert.pem",
        "-inkey",
        "key.pem",
        "-out",
        "decrypted",
    )
    assert run(*decrypt).returncode == 0
    assert (tmp_path / "decrypted").read_bytes() == b"fixture market rows"
    data = bytearray((tmp_path / "transport.cms").read_bytes())
    data[-1] ^= 1
    (tmp_path / "transport.cms").write_bytes(data)
    assert run(*decrypt).returncode != 0


def test_public_workflow_uploads_only_ciphertext():
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/workflows/breeze-history-archive.yml").read_text()
    upload = source.split("uses: actions/upload-artifact@v4", 1)[1]
    assert "breeze-export/" in upload
    assert "breeze-private" not in upload
    assert "-aes-256-gcm" in source
    assert "if: github.ref == 'refs/heads/main'" in source
    assert (
        "PRIVATE KEY"
        not in (root / "deploy/breeze_transport_recipient.pem").read_text()
    )
