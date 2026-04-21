from __future__ import annotations

from pathlib import Path


def test_live_path_files_do_not_use_dynamic_logging_import() -> None:
    files = [
        "src/nifty_scalper_bot/core/app.py",
        "src/nifty_scalper_bot/data/rest/zerodha_client.py",
        "src/nifty_scalper_bot/data/market_data_manager.py",
        "src/nifty_scalper_bot/streaming/websocket_manager.py",
        "src/nifty_scalper_bot/streaming/polling_streamer.py",
        "src/nifty_scalper_bot/strategies/runner.py",
        "src/nifty_scalper_bot/core/strategy_manager.py",
        "src/nifty_scalper_bot/execution/order_manager.py",
        "src/nifty_scalper_bot/execution/bracket_manager.py",
        "src/nifty_scalper_bot/risk/risk_manager.py",
    ]
    needle = '__import__("logging").getLogger(__name__)'
    offenders = [
        file
        for file in files
        if needle in Path(file).read_text(encoding="utf-8")
    ]
    assert offenders == []

