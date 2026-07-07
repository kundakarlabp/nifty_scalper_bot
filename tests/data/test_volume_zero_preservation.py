from zoneinfo import ZoneInfo

from nifty_scalper_bot.data.validator import validate_tick


IST = ZoneInfo("Asia/Kolkata")


def test_validate_tick_preserves_explicit_zero_volume() -> None:
    tick = validate_tick(
        {
            "symbol": "NFO:NIFTY26MAY23500CE",
            "timestamp": "2026-05-13T08:27:25Z",
            "ltp": 300,
            "volume": 0,
            "volume_traded": 6008275,
        }
    )
    assert tick.volume == 0


def test_validate_tick_prefers_volume_delta_zero() -> None:
    tick = validate_tick(
        {
            "symbol": "NFO:NIFTY26MAY23500CE",
            "timestamp": "2026-05-13T08:27:25Z",
            "ltp": 300,
            "volume_delta": 0,
            "volume": 0,
            "volume_traded": 6008275,
        }
    )
    assert tick.volume == 0


def test_validate_tick_prefers_volume_delta_value() -> None:
    tick = validate_tick(
        {
            "symbol": "NFO:NIFTY26MAY23500CE",
            "timestamp": "2026-05-13T08:27:25Z",
            "ltp": 300,
            "volume_delta": 65,
            "volume_traded": 6008275,
        }
    )
    assert tick.volume == 65


def test_validate_tick_converts_aware_timestamp_to_ist() -> None:
    tick = validate_tick(
        {
            "symbol": "NFO:NIFTY26MAY23500CE",
            "timestamp": "2026-05-13T08:27:25Z",
            "ltp": 300,
        }
    )

    assert tick.timestamp.isoformat() == "2026-05-13T13:57:25+05:30"
    assert tick.timestamp.tzinfo == IST


def test_validate_tick_localizes_naive_timestamp_as_ist() -> None:
    tick = validate_tick(
        {
            "symbol": "NFO:NIFTY26MAY23500CE",
            "timestamp": "2026-05-13 08:27:25",
            "ltp": 300,
        }
    )

    assert tick.timestamp.isoformat() == "2026-05-13T08:27:25+05:30"
    assert tick.timestamp.tzinfo == IST
