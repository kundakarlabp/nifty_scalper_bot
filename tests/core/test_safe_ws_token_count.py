from types import SimpleNamespace
from nifty_scalper_bot.core.app import _safe_ws_token_count


def test_safe_ws_token_count_ignores_token_to_symbol():
    mdm = SimpleNamespace(_token_to_symbol={i: f'S{i}' for i in range(54651)}, _subscribed_tokens=set(range(16)))
    ctx = SimpleNamespace(market_data_manager=mdm, websocket_manager=None, broker_client=None)
    assert _safe_ws_token_count(ctx) == 16
