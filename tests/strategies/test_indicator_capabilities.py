from nifty_scalper_bot.strategies.indicators import supports_feature


def test_indicator_capabilities_matrix() -> None:
    assert supports_feature('vwap') is True
    assert supports_feature('bollinger') is True
    assert supports_feature('orb_levels') is True
    assert supports_feature('cpr_levels') is False
    assert supports_feature('structure_flags') is False
    assert supports_feature('order_book_depth') is True
