"""EOD archive preserves dated facts and never supplies executable quotes."""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import date, datetime

import pytest
from scripts.build_eod_contract_db import SCHEMA, check_sha, parse_catalog
from scripts.eod_research_reader import EodReferenceReader, EvidenceUnavailable


def _one_day_catalog():
    row = {
        "TckrSymb": "NIFTY",
        "BizDt": "2025-02-03",
        "FinInstrmTp": "IDO",
        "OptnTp": "CE",
        "XpryDt": "2025-02-06",
        "StrkPric": "22500.00",
        "FinInstrmId": "12345",
        "FinInstrmNm": "NIFTY25FEB22500CE",
        "NewBrdLotQty": "75",
        "OpnPric": "100.00",
        "HghPric": "102.00",
        "LwPric": "99.00",
        "ClsPric": "101.00",
        "LastPric": "101.00",
        "SttlmPric": "101.00",
        "PrvsClsgPric": "98.00",
        "TtlTradgVol": "10",
        "TtlNbOfTxsExctd": "2",
        "OpnIntrst": "100",
        "TtlTrfVal": "1010.00",
        "UndrlygPric": "22500.00",
    }
    return {
        "day": "2025-02-03",
        "available_not_before": "2025-02-04",
        "ex_post_reference_only": True,
        "source_sha256": "f" * 64,
        "source_url": "https://nseindia.com/example",
        "all_nifty_eod_rows": [row],
        "option_identities": [["2025-02-06", 22500, "CE"]],
        "future_expiries": [],
    }


def test_catalog_expost_dated_and_verified():
    summary, rows = parse_catalog(
        gzip.compress(json.dumps(_one_day_catalog()).encode()), "main"
    )
    assert summary["day"] == "2025-02-03"
    assert summary["nextday"] == "2025-02-04"
    assert summary["options"] == 1
    assert rows[0][3] == "CE"
    assert rows[0][6] == 75


def test_catalog_rejects_lookahead_and_unlisted_strikes():
    v = _one_day_catalog()
    v["available_not_before"] = "2025-02-03"
    with pytest.raises(ValueError, match="availability_or_evidence_incorrect"):
        parse_catalog(gzip.compress(json.dumps(v).encode()), "main")
    v["available_not_before"] = "2025-02-04"
    v["option_identities"] = []
    with pytest.raises(ValueError, match="option_catalog_identity_list_mismatch"):
        parse_catalog(gzip.compress(json.dumps(v).encode()), "main")


def test_outer_checksum_rejects_corrupt_bytes(tmp_path):
    p = tmp_path / "test.tar.gz"
    p.write_bytes(b"not a real archive")
    s = tmp_path / "test.tar.gz.sha256"
    s.write_text("0" * 64 + "  test.tar.gz\n")
    with pytest.raises(ValueError, match="source_archive_sha_mismatch"):
        check_sha(p, s)


def test_reader_expost_and_no_broker_quote(tmp_path):
    d = tmp_path / "example.sqlite"
    with sqlite3.connect(d) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,1)",
            (
                "2025-02-03",
                "2025-02-04",
                "e" * 64,
                "https://nseindia.com",
                "main",
                1,
                1,
                0,
            ),
        )
        conn.execute(
            "INSERT INTO contract_eod VALUES (" + ",".join(["?"] * 20) + ")",
            (
                "2025-02-03",
                "12345",
                "NIFTY25FEB22500CE",
                "CE",
                "2025-02-06",
                22500.0,
                75,
                100,
                102,
                99,
                101,
                101,
                101,
                98,
                10,
                2,
                100,
                1010,
                22500,
                0,
            ),
        )
        conn.execute(
            "INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "2025-02-03",
                "2025-02-04",
                22500,
                22600,
                22400,
                22550,
                90000,
                "fixture",
                "main",
            ),
        )
    reader = EodReferenceReader(d)
    day = date(2025, 2, 3)
    with pytest.raises(EvidenceUnavailable, match="same_day_eod_reference_leakage"):
        reader.reference_contracts(
            asof=datetime.fromisoformat("2025-02-03T14:00:00+05:30"), day=day, kind="CE"
        )
    contracts = reader.reference_contracts(
        asof=datetime.fromisoformat("2025-02-04T09:30:00+05:30"), day=day, kind="CE"
    )
    assert len(contracts) == 1
    assert contracts[0]["executable_quote"] is False
    assert (
        reader.daily_index_bars(
            asof=datetime.fromisoformat("2025-02-03T14:00:00+05:30")
        )
        == []
    )
    assert (
        len(
            reader.daily_index_bars(
                asof=datetime.fromisoformat("2025-02-04T09:30:00+05:30")
            )
        )
        == 1
    )
    assert not reader.readiness()["execution_grade_ready"]
    with pytest.raises(
        EvidenceUnavailable, match="historical_bid_ask_and_depth_unavailable"
    ):
        reader.executable_quotes()
    with pytest.raises(
        EvidenceUnavailable, match="historical_executable_fills_unavailable"
    ):
        reader.simulated_fills()
