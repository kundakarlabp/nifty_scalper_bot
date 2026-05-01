from nifty_scalper_bot.core.app import _coerce_ohlc_row


def test_coerce_ohlc_row_mapping() -> None:
    row = _coerce_ohlc_row({'date': '2026-05-01T09:15:00+05:30', 'open': 1, 'high': 2, 'low': 1, 'close': 2, 'volume': 10})
    assert row is not None
    assert row['open'] == 1.0


def test_coerce_ohlc_row_sequence() -> None:
    row = _coerce_ohlc_row(['2026-05-01 09:15:00', 1, 2, 1, 2, 10])
    assert row is not None
    assert row['volume'] == 10


def test_coerce_ohlc_row_invalid() -> None:
    assert _coerce_ohlc_row(['bad']) is None
