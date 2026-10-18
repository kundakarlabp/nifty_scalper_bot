"""FYERS expired option ingestion must preserve provider provenance and abstain."""

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pytest
from scripts.archive_fyers_expired import (
    collect_expired_pair,
    discover_pair,
    normalize_option_candles,
    stage_pair_with_verified_context,
)

IST = ZoneInfo("Asia/Kolkata")
EXPIRY = dt.date(2025, 3, 27)
DAY = dt.date(2025, 3, 26)


def epoch(minute=30):
    start = dt.datetime(2025, 3, 26, 9, 0, tzinfo=IST)
    return int((start + dt.timedelta(minutes=minute)).timestamp())


class FakeFyers:
    def __init__(self, *, bars=2, missing=None):
        self.calls = []
        self.bars = bars
        self.missing = missing

    def expiry_dates(self, args):
        self.calls.append(("expiry", args))
        return {
            "s": "ok",
            "data": {"expiry_dates": {"options": [EXPIRY.isoformat()]}},
        }

    def history_underlying_symbols(self, args):
        self.calls.append(("contracts", args))
        return {
            "s": "ok",
            "data": {
                "expiry_date": EXPIRY.isoformat(),
                "contracts": {
                    "options": [
                        "NSE:NIFTY25MAR23000CE",
                        "NSE:NIFTY25MAR23000PE",
                        "NSE:NIFTY25MAR23100CE",
                    ]
                },
            },
        }

    def fno_historical_data(self, args):
        self.calls.append(("candles", args))
        assert args["resolution"] == "1"
        assert args["include_oi"] == "1"
        assert args["include_greeks"] == "0"
        if self.missing and args["symbol"].endswith(self.missing):
            return {
                "s": "no_data",
                "symbol": args["symbol"],
                "resolution": "1",
                "columns": [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                    "iv",
                ],
                "candles": [],
            }
        rows = [
            [epoch(i + 30), 100, 105, 95, 101, 100, None, None]
            for i in range(self.bars)
        ]
        return {
            "s": "ok",
            "symbol": args["symbol"],
            "resolution": "1",
            "columns": [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
                "iv",
            ],
            "candles": rows,
        }


def archive(client, tmp_path, **overrides):
    return collect_expired_pair(
        client,
        expiry=EXPIRY,
        strike=23000,
        start=DAY,
        end=DAY,
        outdir=tmp_path,
        timestamp_convention="bar_start",
        **overrides,
    )


def test_verified_discovery_and_one_minute_provider_archive_with_nullable_oi(tmp_path):
    client = FakeFyers()
    result = archive(client, tmp_path)
    assert result["promotion_eligible"] is False
    assert result["historical_atm_selection_verified"] is False
    assert result["historical_bid_ask_depth_available"] is False
    assert result["intraday_requests_made"] == 2
    assert result["contracts"]["CE"]["bars"] == 2
    assert [kind for kind, _ in client.calls] == [
        "expiry",
        "contracts",
        "candles",
        "candles",
    ]
    stored = json.loads((tmp_path / result["contracts"]["CE"]["path"]).read_text())
    assert stored["symbol"] == "NSE:NIFTY25MAR23000CE"
    assert stored["source"] == "fyers_expired_fo"
    assert stored["candles"][0][0] == "2025-03-26T09:30:00+05:30"
    assert stored["candles"][0][-1] is None
    # The same provider-verified immutable candles are reused without another
    # intraday API request (important for the Standard-plan 10/day quota).
    cached = archive(client, tmp_path)
    assert cached["intraday_requests_made"] == 0
    assert all(value["status"] == "cached" for value in cached["contracts"].values())
    assert len([kind for kind, _ in client.calls if kind == "candles"]) == 2


def test_valid_no_data_is_reported_without_fabricating_a_candle(tmp_path):
    result = archive(FakeFyers(missing="PE"), tmp_path)
    assert result["contracts"]["PE"] == {"bars": 0, "status": "no_data"}
    assert result["contracts"]["CE"]["bars"] == 2
    assert len(list((tmp_path / "fyers_expired").rglob("*.json"))) == 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("s", "error", "fyers_historical_data_unavailable"),
        (
            "symbol",
            "NSE:NIFTY25MAR23000PE",
            "fyers_candle_contract_or_resolution_mismatch",
        ),
        ("resolution", "5", "fyers_candle_contract_or_resolution_mismatch"),
        ("columns", ["timestamp", "low"], "fyers_candle_columns_invalid"),
        (
            "candles",
            [[epoch(), 99, 94, 95, 99, 100, 1, None]],
            "fyers_candle_values_invalid",
        ),
        (
            "candles",
            [[epoch(), 100, 105, 95, 101, -1, 1, None]],
            "fyers_candle_values_invalid",
        ),
        (
            "candles",
            [[epoch(), 100, 105, 95, 101, 1, -1, None]],
            "fyers_open_interest_invalid",
        ),
        ("candles", [[epoch(), 100, 105, 95, 101, 1]], "fyers_candle_shape_invalid"),
        (
            "candles",
            [
                [epoch(31), 100, 105, 95, 101, 1, 2, None],
                [epoch(30), 100, 105, 95, 101, 1, 2, None],
            ],
            "fyers_candle_timestamp_invalid",
        ),
    ],
)
def test_provider_schema_and_candle_integrity_fail_closed(field, value, error):
    response = FakeFyers().fno_historical_data(
        {
            "resolution": "1",
            "include_oi": "1",
            "include_greeks": "0",
            "symbol": "NSE:NIFTY25MAR23000CE",
        }
    )
    response[field] = value
    with pytest.raises(ValueError, match=error):
        normalize_option_candles(
            response,
            symbol="NSE:NIFTY25MAR23000CE",
            expiry=EXPIRY,
            start=DAY,
            end=DAY,
            timestamp_convention="bar_start",
        )


def test_bar_end_is_shifted_once_and_never_guessed():
    response = FakeFyers().fno_historical_data(
        {
            "resolution": "1",
            "include_oi": "1",
            "include_greeks": "0",
            "symbol": "NSE:NIFTY25MAR23000CE",
        }
    )
    output = normalize_option_candles(
        response,
        symbol="NSE:NIFTY25MAR23000CE",
        expiry=EXPIRY,
        start=DAY,
        end=DAY,
        timestamp_convention="bar_end",
    )
    assert output[0][0] == "2025-03-26T09:29:00+05:30"
    with pytest.raises(ValueError, match="fyers_timestamp_convention_unverified"):
        normalize_option_candles(
            response,
            symbol="NSE:NIFTY25MAR23000CE",
            expiry=EXPIRY,
            start=DAY,
            end=DAY,
            timestamp_convention="unknown",
        )


def test_unverified_expiry_missing_pair_and_bad_strikes_are_not_guessed():
    client = FakeFyers()
    assert discover_pair(client, EXPIRY, 23000)["PE"] == "NSE:NIFTY25MAR23000PE"
    with pytest.raises(ValueError, match="fyers_expired_pair_missing_or_ambiguous"):
        discover_pair(client, EXPIRY, 23100)
    with pytest.raises(ValueError, match="fyers_strike_invalid"):
        discover_pair(client, EXPIRY, 23001)

    class WrongExpiry(FakeFyers):
        def expiry_dates(self, args):
            return {"s": "ok", "data": {"expiry_dates": {"options": []}}}

    with pytest.raises(ValueError, match="fyers_option_expiry_not_listed"):
        discover_pair(WrongExpiry(), EXPIRY, 23000)


def test_tampered_cache_is_not_accepted_and_no_trades_are_sent(tmp_path):
    client = FakeFyers()
    result = archive(client, tmp_path)
    path = tmp_path / result["contracts"]["CE"]["path"]
    raw = json.loads(path.read_text())
    raw["candles"][0][4] = 4999
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fyers_historical_archive_conflict"):
        archive(client, tmp_path)
    assert not any("order" in kind for kind, _ in client.calls)


def test_budget_and_date_checks_prevent_unbounded_calls(tmp_path):
    client = FakeFyers()
    with pytest.raises(ValueError, match="fyers_intraday_request_budget_invalid"):
        archive(client, tmp_path, max_intraday_requests=1)
    assert client.calls == []
    with pytest.raises(ValueError, match="fyers_historical_date_range_invalid"):
        collect_expired_pair(
            client,
            expiry=EXPIRY,
            strike=23000,
            start=DAY,
            end=EXPIRY + dt.timedelta(days=1),
            outdir=tmp_path,
            timestamp_convention="bar_start",
        )
    assert client.calls == []


def _context_archive(tmp_path, count=65):
    directory = tmp_path / "existing_context" / "candles"
    directory.mkdir(parents=True)
    for symbol, kind, expiry in (
        ("NSE:NIFTY 50", "EQ", None),
        ("NFO:NIFTY25MARFUT", "FUT", "2025-03-27"),
    ):
        rows = [
            [
                (
                    dt.datetime(2025, 3, 26, 9, 30, tzinfo=IST)
                    + dt.timedelta(minutes=index)
                ).isoformat(),
                23000,
                23020,
                22980,
                23005,
                250,
            ]
            for index in range(count)
        ]
        (directory / f"{kind}.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "timestamp_convention": "bar_start",
                    "instrument": {
                        "name": "NIFTY",
                        "instrument_type": kind,
                        "expiry": expiry,
                        "lot_size": 75,
                    },
                    "candles": rows,
                }
            )
        )
    return directory.parent


def test_imported_pair_bridges_into_existing_research_loader_only_with_context(
    tmp_path,
):
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    report = archive(FakeFyers(bars=65), tmp_path)
    context = _context_archive(tmp_path)
    destination = tmp_path / "staged" / DAY.isoformat()
    staged = stage_pair_with_verified_context(
        tmp_path,
        context,
        destination,
        day=DAY,
        expiry=EXPIRY,
        strike=23000,
        lot_size=75,
        lot_size_source="operator-attested historical NSE lot-size circular",
        min_synchronized_minutes=60,
    )
    assert staged["promotion_eligible"] is False
    assert staged["historical_atm_selection_verified"] is False
    assert staged["historical_order_book_depth_available"] is False
    histories, instruments, digest = load_archive(destination)
    assert len(histories) == 4
    assert len(histories["NFO:NIFTY25MAR23000CE"]) == 65
    assert len(histories["NFO:NIFTY25MAR23000PE"]) == 65
    assert instruments["NFO:NIFTY25MAR23000CE"]["lot_size"] == 75
    assert staged["data_sha256"] == digest
    assert report["historical_atm_selection_verified"] is False
    with pytest.raises(ValueError, match="fyers_research_stage_prerequisites_invalid"):
        stage_pair_with_verified_context(
            tmp_path,
            context,
            destination,
            day=DAY,
            expiry=EXPIRY,
            strike=23000,
            lot_size=75,
            lot_size_source="verified",
        )


def test_incomplete_context_and_unverified_historical_lot_size_fail_closed(tmp_path):
    archive(FakeFyers(bars=65), tmp_path)
    context = _context_archive(tmp_path, count=3)
    destination = tmp_path / "stage"
    with pytest.raises(ValueError, match="fyers_research_stage_prerequisites_invalid"):
        stage_pair_with_verified_context(
            tmp_path,
            context,
            destination,
            day=DAY,
            expiry=EXPIRY,
            strike=23000,
            lot_size=0,
            lot_size_source="",
        )
    with pytest.raises(
        ValueError, match="fyers_research_stage_synchronized_minutes_missing"
    ):
        stage_pair_with_verified_context(
            tmp_path,
            context,
            destination,
            day=DAY,
            expiry=EXPIRY,
            strike=23000,
            lot_size=75,
            lot_size_source="operator verified",
            min_synchronized_minutes=60,
        )
    assert not destination.exists()


def test_option_payload_tampering_cannot_be_staged(tmp_path):
    report = archive(FakeFyers(bars=65), tmp_path)
    context = _context_archive(tmp_path)
    option_file = tmp_path / report["contracts"]["CE"]["path"]
    payload = json.loads(option_file.read_text())
    payload["candles"][0][4] = 500
    option_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fyers_research_stage_source_untrusted"):
        stage_pair_with_verified_context(
            tmp_path,
            context,
            tmp_path / "staged",
            day=DAY,
            expiry=EXPIRY,
            strike=23000,
            lot_size=75,
            lot_size_source="operator verified",
        )


def test_imported_bar_end_labels_are_normalized_before_research_stage(tmp_path):
    client = FakeFyers(bars=65)
    report = collect_expired_pair(
        client,
        expiry=EXPIRY,
        strike=23000,
        start=DAY,
        end=DAY,
        outdir=tmp_path,
        timestamp_convention="bar_end",
    )
    record = json.loads((tmp_path / report["contracts"]["CE"]["path"]).read_text())
    assert record["source_timestamp_convention"] == "bar_end"
    assert record["timestamp_convention"] == "bar_start"
    assert record["candles"][0][0] == "2025-03-26T09:29:00+05:30"
