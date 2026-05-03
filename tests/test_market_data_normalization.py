from datetime import datetime, timezone

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
