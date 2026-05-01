from pathlib import Path


def test_message_bus_wired_before_websocket_start() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text()
    assert text.index('_wire_and_start_message_bus(ctx)') < text.index('start_websocket()')
