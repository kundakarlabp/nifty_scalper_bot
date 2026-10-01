"""Public NSE download/parse behavior across the bhavcopy format transition."""

import datetime as dt
import io
import zipfile
from types import SimpleNamespace

import pandas as pd
from scripts.sync_nifty_options import fetch_zip_to, main, parse_zip_to_df, url_for


def test_current_bhavcopy_uses_udiff_archive() -> None:
    urls = url_for(dt.date(2026, 9, 30))
    assert urls[0] == (
        "https://archives.nseindia.com/content/fo/"
        "BhavCopy_NSE_FO_0_0_0_20260930_F_0000.csv.zip"
    )
    assert "fo05JUL2024bhav.csv.zip" in url_for(dt.date(2024, 7, 5))[0]


def test_udiff_nifty_options_preserve_contract_and_daily_price_identity(tmp_path):
    rows = pd.DataFrame(
        {
            "TradDt": ["2026-09-30"] * 3,
            "FinInstrmTp": ["IDO", "IDF", "IDO"],
            "TckrSymb": ["NIFTY", "NIFTY", "BANKNIFTY"],
            "XpryDt": ["2026-10-06"] * 3,
            "StrkPric": [25000, 0, 55000],
            "OptnTp": ["CE", "", "PE"],
            "OpnPric": [100, 25000, 200],
            "HghPric": [110, 25100, 220],
            "LwPric": [90, 24900, 180],
            "ClsPric": [105, 25050, 210],
            "SttlmPric": [105, 25050, 210],
            "OpnIntrst": [750, 75, 300],
            "ChngInOpnIntrst": [75, 0, 30],
            "TtlTradgVol": [1500, 75, 600],
            "TtlTrfVal": [157500, 1878750, 126000],
            "NewBrdLotQty": [75] * 3,
        }
    )
    path = tmp_path / "current.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "BhavCopy_NSE_FO_0_0_0_20260930_F_0000.csv", rows.to_csv(index=False)
        )
    result = parse_zip_to_df(str(path))
    assert result is not None
    assert len(result) == 1
    row = result.iloc[0]
    assert row["TIMESTAMP"] == dt.date(2026, 9, 30)
    assert row["EXPIRY_DT"] == dt.date(2026, 10, 6)
    assert row["INSTRUMENT"] == "OPTIDX"
    assert row["STRIKE_PR"] == 25000
    assert row["OPTION_TYP"] == "CE"
    assert row["CLOSE"] == 105
    assert row["CONTRACTS"] == 1500
    assert row["OPEN_INT"] == 750
    assert row["VAL_INLAKH"] == 1.575


def test_corrupt_cache_is_replaced_only_by_a_valid_archive(tmp_path):
    path = tmp_path / "cached.zip"
    path.write_bytes(b"PK truncated")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fo30SEP2026bhav.csv", "INSTRUMENT,SYMBOL\nOPTIDX,NIFTY\n")
    valid_bytes = buffer.getvalue()

    class Session:
        def get(self, *args, **kwargs):
            return SimpleNamespace(status_code=200, content=valid_bytes)

    assert fetch_zip_to(str(path), ["https://example.test/zip"], Session())
    assert path.read_bytes() == valid_bytes


def test_incremental_sync_preserves_previous_contract_days(tmp_path):
    old = pd.DataFrame(
        {
            "TIMESTAMP": ["2026-09-29"],
            "INSTRUMENT": ["OPTIDX"],
            "SYMBOL": ["NIFTY"],
            "EXPIRY_DT": ["2026-10-06"],
            "EXPIRY_CLASS": ["WEEKLY"],
            "STRIKE_PR": [25000],
            "OPTION_TYP": ["CE"],
            "CLOSE": [100],
        }
    )
    old.to_csv(tmp_path / "nifty_options_all.csv", index=False)
    cache = tmp_path / "cache" / "fo" / "2026" / "SEP"
    cache.mkdir(parents=True)
    with zipfile.ZipFile(cache / "fo30SEP2026bhav.csv.zip", "w") as archive:
        archive.writestr(
            "fo30SEP2026bhav.csv",
            (
                "TIMESTAMP,INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,CLOSE\n"
                "30-Sep-2026,OPTIDX,NIFTY,06-Oct-2026,25000,CE,105\n"
            ),
        )
    args = ["--days", "1", "--end-date", "2026-09-30", "--outdir", str(tmp_path)]
    main(args)
    main(args)
    result = pd.read_csv(tmp_path / "nifty_options_all.csv")
    assert list(result["TIMESTAMP"]) == ["2026-09-29", "2026-09-30"]
    assert len(result) == 2
