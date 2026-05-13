from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.normalizers import normalize_history_row


def test_option_history_volume_is_clamped(monkeypatch) -> None:
    monkeypatch.setenv('OPTION_MAX_REASONABLE_1M_VOLUME', '5000000')
    row = {
        'timestamp': datetime(2026, 1, 1, tzinfo=timezone.utc),
        'open': 10,
        'high': 12,
        'low': 9,
        'close': 11,
        'volume': 120319745,
    }
    out = normalize_history_row('NFO:NIFTY26MAY23500CE', row)
    assert out is not None
    assert out['volume'] == 0


def test_non_option_history_volume_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv('OPTION_MAX_REASONABLE_1M_VOLUME', '5000000')
    row = {
        'timestamp': datetime(2026, 1, 1, tzinfo=timezone.utc),
        'open': 10,
        'high': 12,
        'low': 9,
        'close': 11,
        'volume': 120319745,
    }
    out = normalize_history_row('NSE:NIFTY', row)
    assert out is not None
    assert out['volume'] == 120319745
