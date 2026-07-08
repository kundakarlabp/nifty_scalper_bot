from __future__ import annotations


def test_flat_alias_registered_for_operator_commands() -> None:
    import nifty_scalper_bot.notifications  # noqa: F401 - triggers alias patch
    from nifty_scalper_bot.notifications.operator_telegram import OPERATOR_COMMAND_NAMES

    assert "flatten" in OPERATOR_COMMAND_NAMES
    assert "flat" in OPERATOR_COMMAND_NAMES
