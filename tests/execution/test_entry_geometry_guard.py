from types import SimpleNamespace

from nifty_scalper_bot.execution.order_entry_guard_patch import _entry_geometry_block_reason


def test_entry_geometry_blocks_low_reward_to_risk():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=223.90,
        stop_loss=214.61,
        take_profit=227.76,
        intent="ENTRY",
    )

    assert reason is not None
    assert reason["block_reason"] == "entry_rr_below_floor"
    assert reason["rr"] < reason["rr_floor"]


def test_entry_geometry_allows_good_reward_to_risk():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=140.55,
        stop_loss=135.50,
        take_profit=151.00,
        intent="ENTRY",
    )

    assert reason is None


def test_entry_geometry_does_not_block_protective_exit():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        price=135.50,
        stop_loss=None,
        take_profit=None,
        intent="EXIT",
    )

    assert reason is None
