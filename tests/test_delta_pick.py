from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.utils.strike_selector import select_strike_by_delta


def fake_expiry():
    return datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(days=7)


def test_select_strike_by_delta_prefers_nearest():
    chain = [
        {"strike": 20000, "oi": 600000, "median_spread_pct": 0.2},
        {"strike": 20100, "oi": 600000, "median_spread_pct": 0.2},
    ]
    res = select_strike_by_delta(
        spot=20050, opt="CE", expiry=fake_expiry(), target=0.40, band=0.10, chain=chain
    )
    assert res and res["strike"] in (20000, 20100)
