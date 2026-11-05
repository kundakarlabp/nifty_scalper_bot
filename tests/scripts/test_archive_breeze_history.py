"""Breeze research importer must preserve contract identity and integrity."""

import datetime as dt
import json

import pytest
from scripts.archive_breeze_history import collect_pair, normalize

DAY = dt.date(2026, 11, 4)
EXPIRY = dt.date(2026, 11, 10)
STRIKE = 26000


class FakeBreeze:
    def __init__(self, wrong=None):
        self.calls = []
        self.wrong = wrong

    def get_historical_data_v2(self, **kwargs):
        self.calls.append(kwargs)
        side = "Call" if kwargs["right"] == "call" else "Put"
        rows = [
            {
                "datetime": "2026-11-04 09:15:00",
                "stock_code": "NIFTY",
                "exchange_code": "NFO",
                "product_type": "Options",
                "expiry_date": "10-NOV-2026",
                "right": side,
                "strike_price": 26000.0,
                "open": 100,
                "high": 105,
                "low": 99,
                "close": 102,
                "volume": 400,
                "open_interest": None,
            }
        ]
        if self.wrong:
            rows[0][self.wrong[0]] = self.wrong[1]
        return {"Status": 200, "Error": None, "Success": rows}


def archive(client, tmp_path):
    return collect_pair(
        client,
        day=DAY,
        expiry=EXPIRY,
        strike=STRIKE,
        outdir=tmp_path,
        timestamp_convention="bar_start",
    )


def normalize_call(response, convention="bar_start"):
    return normalize(
        response,
        day=DAY,
        expiry=EXPIRY,
        strike=STRIKE,
        side="CE",
        timestamp_convention=convention,
    )


def test_exact_pair_archive_and_deterministic_cache(tmp_path):
    client = FakeBreeze()
    result = archive(client, tmp_path)
    assert result["requests_made"] == 2
    assert result["promotion_eligible"] is False
    assert len(client.calls) == 2
    assert [call["right"] for call in client.calls] == ["call", "put"]
    assert all(call["interval"] == "1minute" for call in client.calls)
    assert result["contracts"]["CE"]["missing_minutes"] == 374
    file = tmp_path / result["contracts"]["CE"]["path"]
    data = json.loads(file.read_text())
    assert data["provider_rows"][0]["open_interest"] is None
    assert data["candles"][0][0] == "2026-11-04T09:15:00+05:30"
    assert data["source_timestamp_convention"] == "bar_start"
    again = archive(client, tmp_path)
    assert again["requests_made"] == 0
    assert again["contracts"]["CE"]["status"] == "cached"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "wrong",
    [
        ("strike_price", 26050),
        ("expiry_date", "17-NOV-2026"),
        ("right", "Put"),
        ("high", 97),
        ("volume", -1),
        ("open_interest", -2),
    ],
)
def test_invalid_contracts_and_bad_prices_fail_closed(tmp_path, wrong):
    with pytest.raises(ValueError, match="breeze_"):
        archive(FakeBreeze(wrong=wrong), tmp_path)


def test_empty_bad_response_duplicate_and_out_of_session():
    good = FakeBreeze().get_historical_data_v2(right="call")
    for response in (
        {"Status": 403, "Error": "invalid", "Success": []},
        {"Status": 200, "Error": None, "Success": "bad"},
    ):
        with pytest.raises(ValueError, match="breeze_provider_error"):
            normalize_call(response)
    for time in (
        "2026-11-04 09:14:00",
        "2026-11-04 15:30:00",
        "2026-11-05 09:15:00",
    ):
        bad = {**good, "Success": [{**good["Success"][0], "datetime": time}]}
        with pytest.raises(ValueError, match="breeze_timestamp_invalid"):
            normalize_call(bad)
    with pytest.raises(ValueError, match="breeze_timestamp_invalid"):
        normalize_call({**good, "Success": good["Success"] * 2})
    candles, rows, missing = normalize_call(
        {"Status": 200, "Error": None, "Success": []}
    )
    assert candles == rows == []
    assert len(missing) == 375


def test_tampered_archive_and_bad_input_never_send_new_requests(tmp_path):
    client = FakeBreeze()
    result = archive(client, tmp_path)
    file = tmp_path / result["contracts"]["CE"]["path"]
    data = json.loads(file.read_text())
    data["candles"][0][4] = 10_000
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="breeze_archive_conflict"):
        archive(client, tmp_path)
    assert len(client.calls) == 2
    with pytest.raises(ValueError, match="breeze_research_contract_or_date_invalid"):
        collect_pair(
            client,
            day=DAY,
            expiry=EXPIRY,
            strike=STRIKE + 1,
            outdir=tmp_path,
            timestamp_convention="bar_start",
        )
    assert len(client.calls) == 2


def test_bar_end_is_explicitly_shifted_once():
    response = FakeBreeze().get_historical_data_v2(right="call")
    response["Success"][0]["datetime"] = "2026-11-04 09:16:00"
    candles, _, missing = normalize_call(response, "bar_end")
    assert candles[0][0] == "2026-11-04T09:15:00+05:30"
    assert len(missing) == 374
    with pytest.raises(ValueError, match="breeze_timestamp_convention_required"):
        normalize_call(response, "unknown")
