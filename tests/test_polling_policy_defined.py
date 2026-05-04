from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from datetime import datetime, timezone


def test_polling_policy_defaults_defined() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    assert mdm._poll_context_stale_seconds > 0  # noqa: SLF001
    assert mdm._poll_option_stale_seconds > 0  # noqa: SLF001
    assert mdm._poll_interval_seconds > 0  # noqa: SLF001
    assert mdm._poll_max_symbols > 0  # noqa: SLF001


def test_should_poll_symbol_no_nameerror() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    assert mdm._should_poll_symbol('NSE:NIFTY', datetime.now(timezone.utc)) in {True, False}  # noqa: SLF001
