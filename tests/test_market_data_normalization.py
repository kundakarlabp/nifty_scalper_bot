from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.normalizers import normalize_history_row


def test_normalize_history_row_list() -> None:
    row = [datetime(2026, 5, 3, 9, 15, tzinfo=timezone.utc), 1, 2, 0.5, 1.5, 100]
    bar = normalize_history_row('NFO:NIFTY26MAY24000CE', row)
    assert bar is not None
    assert bar['symbol'] == 'NFO:NIFTY26MAY24000CE'
    assert bar['open'] == 1.0
    assert bar['close'] == 1.5
    assert bar['volume'] == 100


def test_normalize_history_row_dict() -> None:
    row = {
        'date': datetime(2026, 1, 1),
        'open': 1,
        'high': 2,
        'low': 0.5,
        'close': 1.5,
        'volume': 100,
    }
    bar = normalize_history_row('NSE:NIFTY', row)
    assert bar is not None
    assert bar['close'] == 1.5
    assert bar['source'] == 'historical'


def test_normalize_history_row_invalid() -> None:
    assert normalize_history_row('NSE:NIFTY', {'open': 1}) is None


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_historical_bar_rejects_nonfinite_prices(field, value) -> None:
    row = {
        "date": "2026-09-30T09:30:00+05:30",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "volume": 130,
    }
    row[field] = value
    assert normalize_history_row("NFO:NIFTY26OCT25000CE", row) is None


@pytest.mark.parametrize("shape", ["list", "oi", "open_interest"])
@pytest.mark.parametrize("oi", [0, 650, "650"])
def test_historical_bar_preserves_optional_open_interest(shape, oi) -> None:
    row = ["2026-09-30T09:30:00+05:30", 100, 110, 90, 105, 130, oi]
    if shape != "list":
        row = dict(zip(["date", "open", "high", "low", "close", "volume", shape], row))
    bar = normalize_history_row("NFO:NIFTY26OCT25000CE", row)
    assert bar is not None
    assert bar["oi"] == int(oi)
    assert bar["timestamp"] == datetime(2026, 9, 30, 4, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "oi", [None, "", "invalid", -1, 0.5, float("nan"), float("inf")]
)
def test_missing_or_invalid_optional_oi_does_not_reject_valid_prices(oi) -> None:
    row = ["2026-09-30T09:30:00+05:30", 100, 110, 90, 105, 130, oi]
    bar = normalize_history_row("NFO:NIFTY26OCT25000CE", row)
    assert bar is not None
    assert "oi" not in bar
    assert bar["close"] == 105
